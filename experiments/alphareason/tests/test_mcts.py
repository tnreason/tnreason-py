from dataclasses import dataclass

from experiments.alphareason.actions import AssignValueAction
from experiments.alphareason.mcts import AlphaReasonMCTS


@dataclass
class _ChainState:
    depth: int
    terminal_depth: int = 3

    @property
    def done(self):
        return self.depth >= self.terminal_depth


class _ChainEnv:
    def __init__(self):
        self.step_calls = 0

    def legal_actions(self, state):
        if state.done:
            return []
        return [AssignValueAction(f"x{state.depth}", 0)]

    def step(self, state, action):
        self.step_calls += 1
        next_state = _ChainState(state.depth + 1, state.terminal_depth)
        return next_state, 0.0, next_state.done


class _FlatPolicyValue:
    def action_priors(self, state, legal_actions):
        return {action: 1.0 / len(legal_actions) for action in legal_actions}

    def value(self, state):
        return 1.0 if state.done else 0.0


def test_mcts_without_rollout_only_tests_one_child():
    env = _ChainEnv()
    mcts = AlphaReasonMCTS(
        env=env,
        policy_value=_FlatPolicyValue(),
        simulations=1,
        rollout_depth=0,
    )

    mcts.rank_actions(_ChainState(depth=0))

    assert env.step_calls == 1


def test_mcts_rollout_tests_deeper_path():
    env = _ChainEnv()
    mcts = AlphaReasonMCTS(
        env=env,
        policy_value=_FlatPolicyValue(),
        simulations=1,
        rollout_depth=2,
    )

    mcts.rank_actions(_ChainState(depth=0))

    assert env.step_calls == 3
