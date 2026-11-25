"""tnreason.logic — High-level Symbolic Reasoning and Scaling Gadgets.

This module provides high-level abstractions for mapping relational logic 
to Tensor Networks and structural gadgets for scaling to large domains.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tnreason.engine.subscript_creation import defaultSymbols

class LogicCompiler:
    """A Domain Specific Language (DSL) for Tensor Network Reasoning.
    
    Automates the mapping of First-Order Logic (FOL) predicates to 
    Tensor Network topologies and optimized einsum strings.
    """
    def __init__(self, domain_size: int):
        self.domain_size = domain_size
        self.variables: list[str] = []
        self.constraints: list[tuple[str, tuple[str, ...], np.ndarray]] = []
        self.char_map = defaultSymbols

    def add_variable(self, name: str):
        """Register a variable in the logical system."""
        if name not in self.variables:
            self.variables.append(name)

    def add_constraint(
        self,
        constraint_type: str,
        variables: tuple[str, ...],
        tensor: Optional[np.ndarray] = None,
    ):
        """Add a relational constraint between variables."""
        for v in variables:
            self.add_variable(v)

        if tensor is None:
            if constraint_type == "NEQ":
                if len(variables) != 2:
                    raise ValueError("NEQ constraints require exactly 2 variables.")
                # Binary Not-Equal
                tensor = (1.0 - np.eye(self.domain_size))
            elif constraint_type == "EQ":
                if len(variables) != 2:
                    raise ValueError("EQ constraints require exactly 2 variables.")
                # Binary Equal
                tensor = np.eye(self.domain_size)
            elif constraint_type == "GT":
                if len(variables) != 2:
                    raise ValueError("GT constraints require exactly 2 variables.")
                # Binary Greater-Than
                # Rows index the first variable and columns index the second one.
                tensor = np.tril(np.ones((self.domain_size, self.domain_size)), k=-1)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")
        else:
            tensor = np.asarray(tensor, dtype=float)
            expected_shape = (self.domain_size,) * len(variables)
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"Constraint tensor shape {tensor.shape} does not match "
                    f"{len(variables)} variables over domain size {self.domain_size}."
                )

        self.constraints.append((constraint_type, variables, tensor))

    def compile(self) -> tuple[str, list[np.ndarray]]:
        """Compiles the logical system into an einsum string and core list."""
        if len(self.variables) > len(self.char_map):
            raise ValueError(
                f"Too many variables for einsum compilation: {len(self.variables)} > {len(self.char_map)}."
            )
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
        if not operands:
            return 1.0
        # Semiring-aware einsum (can be extended to custom backends)
        if semiring == "sum_product":
            return float(np.einsum(equation, *operands))
        elif semiring == "max_product":
            # For max-product, we use a simple loop-based contraction for small systems
            # In a real library, this would use a proper semiring-contractor.
            raise NotImplementedError("Advanced semirings require SemiringContractor.")
        elif semiring == "boolean":
            return float(np.einsum(equation, *operands) > 0)
        else:
            raise ValueError(f"Unknown semiring: {semiring}")


class TTGadgets:
    """Structural Tensor-Train decompositions for high-arity constraints."""
    
    @staticmethod
    def alldiff_cores(n: int, d: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Generate sparse transition cores for an N-variable, D-domain AllDiff constraint."""
        num_states = 1 << d
        source_masks = []
        target_masks = []
        for mask in range(num_states):
            for val in range(d):
                bit = 1 << val
                if not (mask & bit):
                    source_masks.append(mask)
                    target_masks.append(mask | bit)
        sparse_core = (
            num_states,
            np.asarray(source_masks, dtype=np.int64),
            np.asarray(target_masks, dtype=np.int64),
        )
        return [sparse_core] * n

    @staticmethod
    def contract_tt(cores: list[tuple[int, np.ndarray, np.ndarray]]) -> float:
        """Sequential contraction of a TT chain."""
        if not cores:
            return 1.0
        state = np.zeros(cores[0][0])
        state[0] = 1.0
        for num_states, source_masks, target_masks in cores:
            if num_states != len(state):
                raise ValueError("All TT cores must share the same state dimension.")
            next_state = np.zeros(num_states)
            np.add.at(next_state, target_masks, state[source_masks])
            state = next_state
        accepting_states = [mask for mask in range(len(state)) if mask.bit_count() == len(cores)]
        return float(state[accepting_states].sum())


class WorldModelUtils:
    """Utilities for scaling TN World Models."""
    
    @staticmethod
    def build_factored_transition(n: int, dims: int = 2) -> list[np.ndarray]:
        """Decompose a grid transition T(s'|s,a) into axis-aligned cores.
        
        Reduces O(S^2) to O(S_i^2).
        """
        if dims != 2:
            raise ValueError("build_factored_transition currently supports exactly 2 dimensions.")
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
    if z_max <= 0:
        raise ValueError("z_max must be positive.")
    return max(0.0, min(1.0, z / z_max))
