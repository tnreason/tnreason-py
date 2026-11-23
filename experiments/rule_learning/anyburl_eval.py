"""Evaluate AnyBURL's rule files with the protocol of boost.py.

AnyBURL is used strictly as a rule generator. Its rules are compiled into sparse score
matrices and handed to the same evaluator, same frozen test set, same tie-break, so the
comparison against our learner is a comparison of rule sets and not of evaluation code.

Rule file format (AnyBURL 23-1x): support \t bodyLength \t confidence \t head <= body1, ...
An atom is rel(arg1,arg2) with single-letter variables and constants such as /m/04ztj.

Only bodies that form a path from X to Y, with at most one constant at an endpoint, are
compiled. The rest are counted and reported, never silently dropped.

Usage:
    python anyburl_eval.py --rules /path/rules-10 --selfcheck
    python anyburl_eval.py --rules /path/rules-10
"""

import argparse
import os
import re
import time

import numpy as np
from scipy import sparse

import boost as B
import kg_rules as K

HERE = os.path.dirname(os.path.abspath(__file__))
ATOM = re.compile(r"([^(),]+)\(([^(),]+),([^(),]+)\)")
VAR = re.compile(r"^[A-Z]$")


def parse_rule(line):
    """support, confidence, headAtom, [bodyAtoms] or None if the line is malformed."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    try:
        support = int(parts[0])
        confidence = float(parts[2])
    except ValueError:
        return None
    head, _, body = parts[3].partition(" <= ")
    headAtoms = ATOM.findall(head)
    bodyAtoms = ATOM.findall(body) if body else []
    if len(headAtoms) != 1:
        return None
    return support, confidence, headAtoms[0], bodyAtoms


def isVar(term):
    return bool(VAR.match(term))


def denseVector(v):
    return np.asarray(v.todense() if sparse.issparse(v) else v).ravel()


def compile_path(bodyAtoms, cores, relIndex, entIndex, n, maxBroadcast=5_000_000):
    """Sparse matrix scoring (X,Y) for a body, or None if it is not compilable.

    Handles the shapes that occur in practice: a path from X to Y, a component that fixes
    one head variable and leaves the other unconstrained (broadcast), and two independent
    components fixing X and Y separately. Anything else is refused and counted.
    """
    varEdges, constFixes = [], []
    for rel, a1, a2 in bodyAtoms:
        if rel not in relIndex:
            return None
        a, b = a1.strip(), a2.strip()
        if isVar(a) and isVar(b):
            varEdges.append((a, b, relIndex[rel]))
        elif isVar(a) or isVar(b):
            var = a if isVar(a) else b
            const = b if isVar(a) else a
            if const not in entIndex:
                return None
            constFixes.append((var, relIndex[rel], entIndex[const], isVar(a)))
        else:
            return None

    # union-find over variables joined by two-variable atoms
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in varEdges:
        parent[find(a)] = find(b)
    for v, _, _, _ in constFixes:
        find(v)
    components = {}
    for v in list(parent):
        components.setdefault(find(v), set()).add(v)

    def componentFactor(vars_):
        """Contract one connected component to a factor over its head variables."""
        fixed = {v: (r, c, aIsVar) for (v, r, c, aIsVar) in constFixes if v in vars_}
        edges = [(a, b, r) for (a, b, r) in varEdges if a in vars_ and b in vars_]
        if not edges:
            if len(vars_) == 1:
                v = next(iter(vars_))
                if v not in fixed:
                    return None
                r, c, aIsVar = fixed[v]
                return ("vec", v, (cores[r][:, c] if aIsVar else cores[r][c, :].T).tocsr())
            return None
        adjacency = {}
        for a, b, r in edges:
            adjacency.setdefault(a, []).append((b, r))
            adjacency.setdefault(b, []).append((a, r))
        if any(len(v) > 2 for v in adjacency.values()):
            return None
        endpoints = [v for v in adjacency if len(adjacency[v]) == 1]
        if len(endpoints) != 2:
            return None
        order, mats, seen = [endpoints[0]], [], {endpoints[0]}
        current = endpoints[0]
        while True:
            candidates = [(w, r) for (w, r) in adjacency[current] if w not in seen]
            if not candidates:
                break
            w, r = candidates[0]
            forward = any((a == current and b == w and rr == r) for (a, b, rr) in edges)
            mats.append((cores[r] if forward else cores[r].T).tocsr())
            order.append(w)
            seen.add(w)
            current = w
        if current not in endpoints or len(seen) != len(vars_):
            return None
        acc = None
        if order[0] in fixed:
            r_, c_, aIsVar_ = fixed[order[0]]
            acc = mats[0][c_, :] if aIsVar_ else mats[0][:, c_].T
            acc = sparse.csr_matrix(acc)
            consumed = 1
        else:
            acc = mats[0]
            consumed = 1
        for i in range(1, len(mats)):
            acc = acc @ mats[i]
        if order[-1] in fixed:
            r_, c_, aIsVar_ = fixed[order[-1]]
            if acc.shape[0] == n and acc.shape[1] == n:
                acc = (acc[:, c_] if not aIsVar_ else acc.T[c_, :].T)
                acc = sparse.csr_matrix(acc)
            else:
                return None
        return ("mat", order[0], order[-1], acc) if acc.shape[0] == n and acc.shape[1] == n \
            else ("vec", order[-1], acc)

    factors = []
    for vs in components.values():
        f = componentFactor(vs)
        if f is None:
            return None
        factors.append(f)

    mat = None
    vec = {}
    for f in factors:
        if f[0] == "mat":
            _, a, b, m = f
            m = m.tocsr()
            if (a, b) == ("X", "Y"):
                pass
            elif (a, b) == ("Y", "X"):
                m = m.T.tocsr()
            else:
                return None
            mat = m if mat is None else mat.multiply(m).tocsr()
        else:
            _, v, vv = f
            if v not in ("X", "Y"):
                return None
            if v in vec:
                vec[v] = vec[v].multiply(vv)
            else:
                vec[v] = vv
    if mat is not None:
        if vec:
            return None
        return mat
    if "X" in vec and "Y" in vec:
        vx, vy = denseVector(vec["X"]), denseVector(vec["Y"])
        rx, cx = np.nonzero(vx)[0], np.nonzero(vy)[0]
        if rx.size * cx.size > maxBroadcast:
            return None
        rows = np.repeat(rx, cx.size)
        cols = np.tile(cx, rx.size)
        data = np.repeat(vx[rx], cx.size) * np.tile(vy[cx], rx.size)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    if "X" in vec or "Y" in vec:
        key = "X" if "X" in vec else "Y"
        v = denseVector(vec[key])
        idx = np.nonzero(v)[0]
        if idx.size * n > maxBroadcast:
            return None
        rows = np.repeat(idx, n)
        cols = np.tile(np.arange(n), idx.size)
        data = np.repeat(v[idx], n)
        if key == "X":
            return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        return sparse.csr_matrix((data, (cols, rows)), shape=(n, n))
    return None


def load_rules(path, cores, relIndex, entIndex, n, maxPerHead=None):
    """Parse first, keep the best maxPerHead per head relation, compile only those."""
    stats = {"lines": 0, "kept": 0, "compiled": 0, "rejected": 0}
    byHead = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parsed = parse_rule(line)
            stats["lines"] += 1
            if parsed is None:
                stats["rejected"] += 1
                continue
            support, confidence, headAtom, bodyAtoms = parsed
            headRel = headAtom[0].strip()
            if headRel not in relIndex or not bodyAtoms:
                stats["rejected"] += 1
                continue
            byHead.setdefault(relIndex[headRel], []).append((confidence, support, bodyAtoms))
    ruleSets = {}
    for h, entries in byHead.items():
        entries.sort(key=lambda e: -e[0])
        if maxPerHead:
            entries = entries[:maxPerHead]
        stats["kept"] += len(entries)
        compiled = []
        for confidence, support, bodyAtoms in entries:
            matrix = compile_path(bodyAtoms, cores, relIndex, entIndex, n)
            if matrix is None:
                stats["rejected"] += 1
                continue
            compiled.append({"matrix": matrix, "confidence": confidence, "support": support})
        if compiled:
            ruleSets[h] = compiled
            stats["compiled"] += len(compiled)
    return ruleSets, stats


def selfcheck():
    """One hand-written chain rule must score the planted composition exactly."""
    rng = np.random.default_rng(3)
    n = 120

    def functional(k=2):
        m = np.zeros((n, n), dtype=bool)
        m[np.arange(n)[:, None], rng.integers(0, n, size=(n, k))] = True
        return m

    a, b = functional(), functional()
    c = (a.astype(np.int32) @ b.astype(np.int32)) > 0
    rows, cols, rIdx = [], [], []
    for mat, rel in ((a, 0), (b, 1), (c, 2)):
        xs, ys = np.nonzero(mat)
        rows.extend(xs.tolist())
        cols.extend(ys.tolist())
        rIdx.extend([rel] * len(xs))
    entities = ["e" + str(i) for i in range(n)]
    relations = ["a", "b", "c"]
    rows, cols, rIdx = np.array(rows), np.array(cols), np.array(rIdx)
    cores = K.relation_cores(rows, cols, rIdx, n, 3)
    relIndex = {r: i for i, r in enumerate(relations)}
    entIndex = {e: i for i, e in enumerate(entities)}

    mat = compile_path([("a", "X", "Z"), ("b", "Z", "Y")], cores, relIndex, entIndex, n)
    assert mat is not None, "chain rule did not compile"
    assert (mat.nnz == c.sum()), "compiled rule covers {} pairs, planted {}".format(mat.nnz, c.sum())
    # atom written with swapped arguments must transpose: a(Z,X) ^ b(Z,Y) is X <- Z -> Y
    reversedMat = compile_path([("a", "Z", "X"), ("b", "Z", "Y")], cores, relIndex, entIndex, n)
    expected = (cores[0].T @ cores[1]).tocsr()
    assert reversedMat is not None and reversedMat.nnz == expected.nnz, "swapped-argument atom mis-compiled"
    refused = compile_path([("a", "X", "Z"), ("b", "W", "Y")], cores, relIndex, entIndex, n)
    assert refused is None, "disconnected body must be refused, not compiled"
    print("selfcheck passed: chain compiles to {} pairs, reversed order matches, disconnected refused".format(
        mat.nnz), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--rules", default="/Users/eigel/work/tools/anyburl/rules-10")
    parser.add_argument("--dir", default=os.path.join(HERE, "data", "fb15k237"))
    parser.add_argument("--maxPerHead", type=int, default=60)
    parser.add_argument("--aggregation", default="max", choices=["max", "sum"])
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        trainPath = os.path.join(args.dir, "train.txt")
        entities, relations, rows, cols, rIdx = K.load_triples(trainPath)
        entIndex = {e: i for i, e in enumerate(entities)}
        relIndex = {r: i for i, r in enumerate(relations)}
        cores = K.relation_cores(rows, cols, rIdx, len(entities), len(relations))
        t0 = time.time()
        ruleSets, stats = load_rules(args.rules, cores, relIndex, entIndex, len(entities),
                                     maxPerHead=args.maxPerHead)
        print("parsed {} lines, kept {} of them, compiled {} rules, {} rejected ({:.1f}s)".format(
            stats["lines"], stats["kept"], stats["compiled"], stats["rejected"], time.time() - t0), flush=True)
        trainIdx = list(zip(rows.tolist(), rIdx.tolist(), cols.tolist()))
        B.evaluate(cores, ruleSets, len(entities), relations, entIndex, relIndex, trainIdx,
                   [os.path.join(args.dir, "valid.txt")], os.path.join(args.dir, "test.txt"),
                   None, False, args.aggregation)
