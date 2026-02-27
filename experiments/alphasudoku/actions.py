from dataclasses import dataclass
from typing import Tuple, Union


def cluster_key_from_colors(colors: Tuple[str, ...]) -> str:
    return "_".join(colors[::-1])


@dataclass(frozen=True)
class ClusterCandidate:
    key: str
    colors: Tuple[str, ...]


@dataclass(frozen=True)
class AssignAction:
    atom_key: str


@dataclass(frozen=True)
class AddClusterAction:
    cluster_key: str


AlphaSudokuAction = Union[AssignAction, AddClusterAction]

