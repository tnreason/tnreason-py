from demonstrations.sudoku.constraints import sudoku_constraints as standard_sudoku
from demonstrations.sudoku.examples import sakana_fish

from .closure_engine import AlphaSudokuClosureEngine
from .inference import get_forward_mp_inferer_from_canetwork
from .validation import validate_assignment


def _smoke_check(canetwork_factory, num=3):
    empty_network = canetwork_factory(num=num, startAssignment={})
    propagator = get_forward_mp_inferer_from_canetwork(empty_network)
    assert propagator.caNetwork is empty_network

    engine = AlphaSudokuClosureEngine(
        num=num,
        canetwork_factory=lambda assignment: canetwork_factory(num=num, startAssignment=assignment),
    )
    state = engine.close({})
    assert not state.contradiction

    valid, score = validate_assignment(empty_network, {})
    assert valid
    assert score == 1


def test_standard_sudoku_backend():
    _smoke_check(standard_sudoku.get_assignment_as_CANetwork, num=3)


def test_sakana_fish_backend():
    _smoke_check(sakana_fish.get_assignment_as_CANetwork, num=3)
