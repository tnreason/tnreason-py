from demonstrations.sudoku.examples import sakana_fish, standard_sudoku
from experiments.constraint_networks.unit_propagation import GenericUnitPropagation

from experiments.alphareason.closure_engine import AlphaReasonClosureEngine
from experiments.alphareason.env import AlphaReasonEnv
from experiments.alphareason.mcts import AlphaReasonMCTS
from experiments.alphareason.tasks.sudoku_examples import (
    build_sudoku_example_task,
    load_standard_example_evidence,
)


def test_loads_standard_example_evidence_without_executing_example():
    evidence = load_standard_example_evidence("n=2")

    assert evidence == {
        "a_0_1_0_0_1": 1,
        "a_0_0_0_1_0": 1,
        "a_0_1_1_0_3": 1,
        "a_1_0_0_0_2": 1,
        "a_0_0_1_0_2": 1,
    }


def test_builds_n3_example_task_from_demonstration_file():
    task = build_sudoku_example_task("n=3")

    assert task.name == "sudoku_example_n=3"
    assert task.metadata == {"num": 3, "example_name": "n=3"}
    assert len(task.initial_evidence) == 30
    assert len(task.target_feature_keys) == 81
    assert task.network_factory(task.initial_evidence).computationCoreDict


def test_builds_sakana_fish_task_from_demonstration_builder():
    task = build_sudoku_example_task("sakana_fish")

    assert task.name == "sudoku_example_sakana_fish"
    assert task.initial_evidence == {}
    assert len(task.target_feature_keys) == 81
    assert task.network_factory(task.initial_evidence).computationCoreDict


def test_n2_example_can_be_used_by_mcts():
    task = build_sudoku_example_task("n=2")
    env = AlphaReasonEnv(task=task, closure_engine=AlphaReasonClosureEngine(max_message_count=500))
    mcts = AlphaReasonMCTS(env=env, simulations=1)

    state = env.reset()
    if not state.done:
        action = mcts.choose_action(state)
        state, reward, done = env.step(state, action)

        assert action is not None
        assert reward >= -1.0
        assert done == state.done

    assert not state.contradiction


def test_standard_sudoku_example_exposes_constraint_network_for_forward_chaining():
    evidence = load_standard_example_evidence("n=2")

    core_dict = standard_sudoku.get_assignment_as_constraint_network(num=2, startAssignment=evidence)
    assert evidence.keys() <= core_dict.keys()

    chainer = GenericUnitPropagation(core_dict)
    chainer.propagate_all_singleNodeEdges()

    assert chainer.consistent
    assert "pos_0_0_0_0_core" in chainer.cn.coresDict


def test_sakana_fish_example_exposes_constraint_network_for_forward_chaining():
    core_dict = sakana_fish.get_assignment_as_constraint_network(num=3, startAssignment={})

    assert "white_dot_pos_0_0_0_0_pos_0_1_0_0" in core_dict
    assert "black_dot_pos_0_1_0_2_pos_0_2_0_2" in core_dict
    assert "red_line_pos_0_2_0_1_pos_0_2_0_2" in core_dict

    chainer = GenericUnitPropagation(core_dict)
    chainer.propagate_all_singleNodeEdges()

    assert chainer.consistent
