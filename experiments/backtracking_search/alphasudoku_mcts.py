
"""alphasudoku_mcts.py

A lightweight single-player MCTS (PUCT) wrapper around tnreason Sudoku forward message passing.

- State is a Sudoku evidence dict: {atom_key: 1, ...} where atom_key like 'a_r1_r2_c1_c2_n'
- An action is setting one additional atom_key to 1 (i.e., guessing a digit for a cell).
- Transition: evidence' = evidence ∪ {atom_key: 1}, then run forward message passing to propagate.

This is designed to plug into enexa-tensor-reasoning-version1 experiments/backtracking_search/backtrack_caNetwork.py
by replacing the random guess with MCTS-based choice.

Tested against the structure of:
  demonstrations/sudoku/sudoku_forward_mp.py  (get_forward_mp_inferer)
and feature naming conventions used in sudoku_constraints.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import math
import random
import re

# --- Utilities for atom naming ------------------------------------------------

_ATOM_RE = re.compile(r"^a_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)$")

def parse_atom(atom_key: str) -> Optional[Tuple[int,int,int,int,int]]:
    """Return (r1,r2,c1,c2,n) if atom key matches, else None."""
    m = _ATOM_RE.match(atom_key)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # type: ignore

def cell_id_from_atom(atom_key: str) -> Optional[Tuple[int,int,int,int]]:
    p = parse_atom(atom_key)
    if p is None:
        return None
    r1,r2,c1,c2,_ = p
    return (r1,r2,c1,c2)

def all_cell_ids(num: int) -> List[Tuple[int,int,int,int]]:
    return [(r1,r2,c1,c2) for r1 in range(num) for r2 in range(num) for c1 in range(num) for c2 in range(num)]

def atom_key(num: int, r1: int, r2: int, c1: int, c2: int, n: int) -> str:
    return f"a_{r1}_{r2}_{c1}_{c2}_{n}"

# --- MP interface -------------------------------------------------------------

class MpBackend:
    """Thin adapter around demonstrations.sudoku.sudoku_forward_mp.get_forward_mp_inferer."""

    def __init__(self, num: int, get_inferer: Callable[[int, Dict[str,int]], object]):
        self.num = num
        self.get_inferer = get_inferer

    def propagate(self, evidence: Dict[str,int], max_message_count: int = 30000, inference_clusters: Optional[Dict[str, List[str]]] = None):
        """Run forward MP. Returns (propagator, contradiction_bool)."""
        propagator = self.get_inferer(self.num, evidence)
        if inference_clusters is not None:
            propagator.update_inferenceClusters(inference_clusters)
        try:
            # tnreason API: propagate_until_convergence(nonTrivialFeatureKeys=..., maxMessageCount=...)
            propagator.propagate_until_convergence(
                nonTrivialFeatureKeys=list(evidence.keys()),
                maxMessageCount=max_message_count,
                verbose=False,
            )
            return propagator, False
        except Exception:
            return propagator, True

# --- Heuristics: candidates, terminal check, value ----------------------------

def evidence_is_solved(num: int, evidence: Dict[str,int]) -> bool:
    """Solved if each cell has exactly one atom assigned to 1."""
    per_cell = {}
    for k,v in evidence.items():
        if v != 1 or not k.startswith("a_"):
            continue
        cid = cell_id_from_atom(k)
        if cid is None:
            continue
        per_cell[cid] = per_cell.get(cid, 0) + 1
        if per_cell[cid] > 1:
            return False
    # must cover all cells
    return len(per_cell) == (num**4)

def forced_atoms_from_meanparams(mean_param_dict: dict) -> Dict[str,int]:
    """Extract atoms that are certainly true according to mean params (prob(0)==0)."""
    out = {}
    for feature_key, core in mean_param_dict.items():
        if isinstance(feature_key, str) and feature_key.startswith("a_"):
            try:
                # core is a dict-like object in tnreason; use the same logic as meanParams_to_evidence
                if core[{feature_key: 0}] == 0:
                    out[feature_key] = 1
            except Exception:
                # fall back: ignore if not indexable that way
                pass
    return out

def cell_candidates_from_canparams(num: int, can_param_dict: dict, evidence: Dict[str,int]) -> Dict[Tuple[int,int,int,int], List[str]]:
    """Return allowed atom-keys (assignable to 1) per cell, based on canParam support."""
    cand: Dict[Tuple[int,int,int,int], List[str]] = {cid: [] for cid in all_cell_ids(num)}
    for key, core in can_param_dict.items():
        if not (isinstance(key, str) and key.startswith("a_")):
            continue
        if key in evidence:
            continue
        cid = cell_id_from_atom(key)
        if cid is None:
            continue
        # HardPartitionFeature canParam is a boolean vector over {0,1}; index 1 means "can be true"
        try:
            if core[{key: 1}] > 0:
                cand[cid].append(key)
        except Exception:
            # If the core isn't directly indexable as dict, ignore
            continue
    return cand

def choose_most_constrained_cell(num: int, cand_per_cell: Dict[Tuple[int,int,int,int], List[str]], evidence: Dict[str,int]) -> Optional[Tuple[int,int,int,int]]:
    """Pick a cell with >1 candidates and minimal candidate count."""
    # skip already decided cells
    decided = set()
    for k,v in evidence.items():
        if v==1 and k.startswith("a_"):
            cid = cell_id_from_atom(k)
            if cid is not None:
                decided.add(cid)
    best = None
    best_k = 10**9
    for cid, cands in cand_per_cell.items():
        if cid in decided:
            continue
        k = len(cands)
        if 1 < k < best_k:
            best_k = k
            best = cid
    return best

def atom_prior_from_meanparams(mean_param_dict: dict, atom: str) -> float:
    """Try to read P(atom=1) from meanParamDict for HardPartitionFeature."""
    try:
        core = mean_param_dict[atom]
        p1 = float(core[{atom: 1}])
        p0 = float(core[{atom: 0}])
        s = p0 + p1
        if s <= 0:
            return 0.0
        return max(0.0, min(1.0, p1 / s))
    except Exception:
        return 0.0

def value_from_cell_entropies(num: int, mean_param_dict: dict, evidence: Dict[str,int]) -> float:
    """Heuristic value in [-1,1] based on how concentrated each cell's digit distribution is."""
    # For each cell, build digit probs p(n)=P(a_cell_n=1) and compute entropy.
    entropies = []
    decided = set()
    for k,v in evidence.items():
        if v==1 and k.startswith("a_"):
            cid = cell_id_from_atom(k)
            if cid is not None:
                decided.add(cid)
    for cid in all_cell_ids(num):
        if cid in decided:
            continue
        r1,r2,c1,c2 = cid
        ps = []
        for n in range(num**2):
            akey = atom_key(num, r1,r2,c1,c2,n)
            ps.append(atom_prior_from_meanparams(mean_param_dict, akey))
        s = sum(ps)
        if s <= 1e-12:
            # looks like contradiction / extremely uncertain; treat as bad
            entropies.append(math.log(num**2))
            continue
        ps = [p/s for p in ps]
        ent = -sum(p*math.log(p + 1e-12) for p in ps)
        entropies.append(ent)
    if not entropies:
        return 1.0
    # normalize entropy: 0 (best) .. log(K) (worst)
    max_ent = math.log(num**2)
    avg_ent = sum(entropies) / len(entropies)
    score01 = 1.0 - (avg_ent / max_ent)   # 1 is good
    return max(-1.0, min(1.0, 2.0*score01 - 1.0))

# --- MCTS --------------------------------------------------------------------

@dataclass
class EdgeStats:
    prior: float
    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

@dataclass
class Node:
    evidence: Dict[str,int]
    # map action(atom_key)->child node
    children: Dict[str, "Node"] = field(default_factory=dict)
    # stats per action
    stats: Dict[str, EdgeStats] = field(default_factory=dict)
    expanded: bool = False
    terminal: bool = False
    terminal_value: float = 0.0  # used if terminal

class AlphaSudokuMCTS:
    """Single-player PUCT MCTS for Sudoku, using tnreason forward MP as evaluator."""

    def __init__(
        self,
        num: int,
        mp_backend: MpBackend,
        c_puct: float = 1.5,
        simulations: int = 400,
        max_message_count: int = 30000,
        top_k_actions: int = 6,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_eps: float = 0.25,
        rng: Optional[random.Random] = None,
    ):
        self.num = num
        self.mp = mp_backend
        self.c_puct = c_puct
        self.simulations = simulations
        self.max_message_count = max_message_count
        self.top_k_actions = top_k_actions
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or random.Random()

    def pick_action(self, root_evidence: Dict[str,int], inference_clusters: Optional[Dict[str, List[str]]] = None) -> str:
        """Run MCTS and return the best next atom_key to set to 1."""
        root = Node(evidence=dict(root_evidence))
        # expand once to get root priors (and optionally inject Dirichlet noise like AlphaZero)
        self._expand(root, add_dirichlet_noise=True, inference_clusters=inference_clusters)

        for _ in range(self.simulations):
            path: List[Tuple[Node, str]] = []
            node = root

            # Selection
            while node.expanded and not node.terminal and node.children:
                action = self._select_puct(node)
                path.append((node, action))
                node = node.children[action]

            # Expansion / Evaluation
            if node.terminal:
                value = node.terminal_value
            else:
                value = self._expand(node, add_dirichlet_noise=False, inference_clusters=inference_clusters)

            # Backup (single-player: no sign flip)
            for parent, action in path:
                st = parent.stats[action]
                st.N += 1
                st.W += value

        # choose most visited action
        if not root.stats:
            raise ValueError("No legal actions from root state.")
        best_action = max(root.stats.items(), key=lambda kv: kv[1].N)[0]
        return best_action

    def _select_puct(self, node: Node) -> str:
        total_N = sum(st.N for st in node.stats.values()) + 1
        sqrt_total = math.sqrt(total_N)
        best_a = None
        best_u = -1e18
        for a, st in node.stats.items():
            u = st.Q + self.c_puct * st.prior * (sqrt_total / (1 + st.N))
            if u > best_u:
                best_u = u
                best_a = a
        assert best_a is not None
        return best_a

    def _expand(self, node: Node, add_dirichlet_noise: bool, inference_clusters: Optional[Dict[str, List[str]]] = None) -> float:
        """Evaluate node with MP, mark terminal, and populate priors/stats/children. Return value estimate."""
        propagator, contradiction = self.mp.propagate(
            node.evidence,
            max_message_count=self.max_message_count,
            inference_clusters=inference_clusters
        )
        if contradiction:
            node.terminal = True
            node.terminal_value = -1.0
            node.expanded = True
            return -1.0

        # absorb forced truths from mean params into evidence (like backtracker's plateau step)
        try:
            forced = forced_atoms_from_meanparams(propagator.meanParamDict)
        except Exception:
            forced = {}
        if forced:
            node.evidence = {**node.evidence, **forced}

        if evidence_is_solved(self.num, node.evidence):
            node.terminal = True
            node.terminal_value = 1.0
            node.expanded = True
            return 1.0

        # compute candidates by most constrained cell
        cand_per_cell = cell_candidates_from_canparams(self.num, propagator.caNetwork.canParamDict, node.evidence)
        cell = choose_most_constrained_cell(self.num, cand_per_cell, node.evidence)
        if cell is None:
            # no >1-candidate cell left; either solved already (handled) or stuck; treat as near-solved
            node.terminal = False
            node.expanded = True
            return value_from_cell_entropies(self.num, propagator.meanParamDict, node.evidence)

        actions = cand_per_cell[cell]

        # priors from mean params (P(atom=1)), fallback uniform over allowed
        priors = [(a, atom_prior_from_meanparams(propagator.meanParamDict, a)) for a in actions]
        # if all priors ~0, make uniform
        s = sum(p for _, p in priors)
        if s <= 1e-12:
            priors = [(a, 1.0) for a, _ in priors]
            s = float(len(priors))
        priors = [(a, p / s) for a, p in priors]

        # top-k pruning
        priors.sort(key=lambda t: t[1], reverse=True)
        priors = priors[: max(1, self.top_k_actions)]
        s2 = sum(p for _, p in priors)
        priors = [(a, p / s2) for a, p in priors]

        if add_dirichlet_noise and self.dirichlet_alpha is not None and len(priors) > 1:
            noise = self._dirichlet([self.dirichlet_alpha] * len(priors))
            priors = [
                (a, (1 - self.dirichlet_eps) * p + self.dirichlet_eps * n)
                for (a, p), n in zip(priors, noise)
            ]
            # renormalize
            s3 = sum(p for _, p in priors)
            priors = [(a, p / s3) for a, p in priors]

        # build children and stats
        node.children = {}
        node.stats = {}
        for a, p in priors:
            child_ev = dict(node.evidence)
            child_ev[a] = 1
            node.children[a] = Node(evidence=child_ev)
            node.stats[a] = EdgeStats(prior=float(p))

        node.expanded = True
        return value_from_cell_entropies(self.num, propagator.meanParamDict, node.evidence)

    def _dirichlet(self, alpha: List[float]) -> List[float]:
        # simple gamma-based Dirichlet sampler (no numpy dependency)
        xs = []
        for a in alpha:
            # Marsaglia & Tsang for gamma(k=a, theta=1) needs numpy; use random.gammavariate
            xs.append(self.rng.gammavariate(a, 1.0))
        s = sum(xs)
        return [x / s for x in xs] if s > 0 else [1.0 / len(xs)] * len(xs)


# --- Convenience: integrate like Backtracker ---------------------------------

def mcts_guess_next_atom(
    num: int,
    evidence: Dict[str,int],
    get_inferer: Callable[[int, Dict[str,int]], object],
    simulations: int = 400,
    top_k_actions: int = 6,
    c_puct: float = 1.5,
) -> str:
    """One-shot helper: run MCTS from evidence and return the next atom to set true."""
    mp = MpBackend(num=num, get_inferer=get_inferer)
    mcts = AlphaSudokuMCTS(
        num=num,
        mp_backend=mp,
        simulations=simulations,
        top_k_actions=top_k_actions,
        c_puct=c_puct,
        # You can enable Dirichlet noise for exploration at root:
        dirichlet_alpha=None,
    )
    return mcts.pick_action(evidence)
