from demonstrations.sudoku.examples import sakana_fish, standard_sudoku

from .closure_engine import AlphaSudokuClosureEngine


def _smoke_check(canetwork_factory, num=3):
    assert callable(canetwork_factory)

    engine = AlphaSudokuClosureEngine(
        num=num,
        canetwork_factory=lambda assignment: canetwork_factory(num=num, startAssignment=assignment),
    )
    assert engine.num == num
    assert callable(engine.canetwork_factory)


def test_standard_sudoku_backend():
    _smoke_check(standard_sudoku.get_assignment_as_CANetwork, num=3)


def test_sakana_fish_backend():
    _smoke_check(sakana_fish.get_assignment_as_CANetwork, num=3)
