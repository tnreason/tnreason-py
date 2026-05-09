"""tnreason.logic — High-level Symbolic Reasoning and Scaling Gadgets.

This module provides high-level abstractions for mapping relational logic 
to Tensor Networks and structural gadgets for scaling to large domains.
"""

from __future__ import annotations
import numpy as np
import string
from typing import List, Tuple, Dict, Any, Optional

class LogicCompiler:
    """A Domain Specific Language (DSL) for Tensor Network Reasoning.
    
    Automates the mapping of First-Order Logic (FOL) predicates to 
    Tensor Network topologies and optimized einsum strings.
    """
    def __init__(self, domain_size: int):
        self.domain_size = domain_size
        self.variables: list[str] = []
        self.constraints: list[tuple[str, tuple[str, ...], np.ndarray]] = []
        self.char_map = string.ascii_lowercase + string.ascii_uppercase

    def add_variable(self, name: str):
        """Register a variable in the logical system."""
        if name not in self.variables:
            self.variables.append(name)

    def add_constraint(self, type: str, vars: tuple[str, ...], tensor: Optional[np.ndarray] = None):
        """Add a relational constraint between variables."""
        for v in vars:
            self.add_variable(v)
        
        if tensor is None:
            if type == "NEQ":
                # Binary Not-Equal
                tensor = (1.0 - np.eye(self.domain_size))
            elif type == "EQ":
                # Binary Equal
                tensor = np.eye(self.domain_size)
            elif type == "GT":
                # Binary Greater-Than
                tensor = np.zeros((self.domain_size, self.domain_size))
                for i in range(self.domain_size):
                    for j in range(self.domain_size):
                        if i > j: tensor[i, j] = 1.0
            else:
                raise ValueError(f"Unknown constraint type: {type}")
        
        self.constraints.append((type, vars, tensor))

    def compile(self) -> tuple[str, list[np.ndarray]]:
        """Compiles the logical system into an einsum string and core list."""
        var_to_char = {v: self.char_map[i] for i, v in enumerate(self.variables)}
        operands = []
        einsum_parts = []
        used_vars = set()
        
        for _, vars, tensor in self.constraints:
            indices = "".join(var_to_char[v] for v in vars)
            einsum_parts.append(indices)
            operands.append(tensor)
            for v in vars: used_vars.add(v)
        
        # Add unary 'ones' factors for unconstrained variables (partition function safety)
        for v in self.variables:
            if v not in used_vars:
                einsum_parts.append(var_to_char[v])
                operands.append(np.ones(self.domain_size))
        
        equation = ",".join(einsum_parts) + "->"
        return equation, operands

    def solve(self, semiring: str = "sum_product") -> float:
        """Solves the system and returns the partition function Z."""
        equation, operands = self.compile()
        # Semiring-aware einsum (can be extended to custom backends)
        if semiring == "sum_product":
            return float(np.einsum(equation, *operands))
        elif semiring == "max_product":
            # For max-product, we use a simple loop-based contraction for small systems
            # In a real library, this would use a proper semiring-contractor.
            raise NotImplementedError("Advanced semirings require SemiringContractor.")
        else:
            return float(np.einsum(equation, *operands) > 0)


class TTGadgets:
    """Structural Tensor-Train decompositions for high-arity constraints."""
    
    @staticmethod
    def alldiff_cores(n: int, d: int) -> list[np.ndarray]:
        """Generate sparse TT cores for an N-variable, D-domain AllDiff constraint.
        
        Complexity: O(N * D * 2^D) non-zeros.
        """
        num_states = 1 << d
        cores = []
        for _ in range(n):
            # Shape: (bond_in, physical, bond_out)
            core = np.zeros((num_states, d, num_states))
            for mask in range(num_states):
                for val in range(d):
                    bit = 1 << val
                    if not (mask & bit):
                        core[mask, val, mask | bit] = 1.0
            cores.append(core)
        return cores

    @staticmethod
    def contract_tt(cores: list[np.ndarray]) -> float:
        """Sequential contraction of a TT chain."""
        state = np.zeros(cores[0].shape[0])
        state[0] = 1.0
        for core in cores:
            state = np.einsum('i,ijk->jk', state, core).sum(axis=0)
        return float(state[-1])


class WorldModelUtils:
    """Utilities for scaling TN World Models."""
    
    @staticmethod
    def build_factored_transition(n: int, dims: int = 2) -> list[np.ndarray]:
        """Decompose a grid transition T(s'|s,a) into axis-aligned cores.
        
        Reduces O(S^2) to O(S_i^2).
        """
        # Example for 4 actions: up, right, down, left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        cores = []
        for d in range(dims):
            # T_d[x', x, a]
            t = np.zeros((n, n, 4))
            for aid, delta in enumerate(moves):
                dr = delta[d]
                for x in range(n):
                    nx = x + dr
                    if 0 <= nx < n: t[nx, x, aid] = 1.0
                    else: t[x, x, aid] = 1.0
            cores.append(t)
        return cores


def consistency_certificate(z: float, z_max: float) -> float:
    """Returns a continuous confidence score in [0, 1]."""
    if z_max == 0: return 0.0
    return min(1.0, z / z_max)
