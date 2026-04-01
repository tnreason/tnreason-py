from dataclasses import dataclass
from typing import Tuple, Union


def cluster_key_from_colors(colors: Tuple[str, ...]) -> str:
    return "_".join(colors[::-1])


@dataclass(frozen=True)
class AssignValueAction:
    feature_key: str
    value_index: int


@dataclass(frozen=True)
class AddClusterAction:
    cluster_key: str


AlphaReasonAction = Union[AssignValueAction, AddClusterAction]

