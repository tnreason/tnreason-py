from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .dataset import SupervisedSample, build_supervised_dataset
from .policies import AlphaReasonPolicyValueNet, action_to_key
from .run_experiment import build_demo_sudoku_task
from .task import AlphaReasonTask


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_samples_per_task: Optional[int] = None
    shuffle_good_actions: bool = True
    seed: int = 0
    checkpoint_path: Optional[str] = None


def infer_model_shape(tasks: Sequence[AlphaReasonTask]) -> tuple[int, int]:
    num_features = max(len(task.target_feature_keys) for task in tasks)
    max_feature_dim = max(max(task.feature_domain_sizes.values()) for task in tasks)
    return num_features, max_feature_dim


def compute_policy_value_loss(
    model: AlphaReasonPolicyValueNet,
    sample: SupervisedSample,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    legal_actions = list(sample.state.legal_actions)
    state_tensor = model.encode_state(sample.state)
    action_tensor = model.encode_actions(sample.state, legal_actions)
    logits, value = model.forward_tensors(state_tensor, action_tensor)

    target_probs = torch.zeros(len(legal_actions), dtype=torch.float32, device=model.device)
    action_index = {action_to_key(action): idx for idx, action in enumerate(legal_actions)}
    for action, prob in sample.policy_target.items():
        key = action_to_key(action)
        if key in action_index:
            target_probs[action_index[key]] = float(prob)

    if target_probs.sum() <= 0:
        target_probs = torch.full_like(target_probs, 1.0 / max(1, len(legal_actions)))
    else:
        target_probs = target_probs / target_probs.sum()

    log_probs = F.log_softmax(logits, dim=0)
    policy_loss = -(target_probs * log_probs).sum()

    value_target = torch.tensor([sample.value_target], dtype=torch.float32, device=model.device)
    value_loss = F.mse_loss(value.view(1), value_target)
    total_loss = policy_loss + value_loss
    return total_loss, policy_loss.detach(), value_loss.detach()


def train_supervised(
    tasks: Sequence[AlphaReasonTask],
    config: Optional[TrainingConfig] = None,
    model: Optional[AlphaReasonPolicyValueNet] = None,
) -> tuple[AlphaReasonPolicyValueNet, List[Dict[str, float]]]:
    if not tasks:
        raise ValueError("train_supervised requires at least one task.")

    config = config or TrainingConfig()
    num_features, max_feature_dim = infer_model_shape(tasks)
    model = model or AlphaReasonPolicyValueNet(
        num_features=num_features,
        max_feature_dim=max_feature_dim,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    dataset = build_supervised_dataset(
        tasks,
        max_samples_per_task=config.max_samples_per_task,
        shuffle_good_actions=config.shuffle_good_actions,
        seed=config.seed,
    )
    if not dataset:
        raise ValueError("No supervised samples could be extracted from the provided tasks.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0

        for sample_idx in permutation:
            sample = dataset[sample_idx]
            optimizer.zero_grad()
            total_loss, policy_loss, value_loss = compute_policy_value_loss(model, sample)
            total_loss.backward()
            optimizer.step()

            total_loss_sum += float(total_loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())

        epoch_stats = {
            "epoch": float(epoch + 1),
            "total_loss": total_loss_sum / len(dataset),
            "policy_loss": policy_loss_sum / len(dataset),
            "value_loss": value_loss_sum / len(dataset),
            "samples": float(len(dataset)),
        }
        history.append(epoch_stats)

    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_features": model.num_features,
                "max_feature_dim": model.max_feature_dim,
                "history": history,
            },
            checkpoint_path,
        )

    return model, history


def build_demo_training_tasks() -> List[AlphaReasonTask]:
    return [
        build_demo_sudoku_task(),
        build_demo_sudoku_task(
            initial="1.3." ".4.." "..4." "4..1",
            name="demo_standard_sudoku_4x4_variant_a",
        ),
        build_demo_sudoku_task(
            initial="..3." "34.." "..4." "4...",
            name="demo_standard_sudoku_4x4_variant_b",
        ),
    ]


if __name__ == "__main__":
    tasks = build_demo_training_tasks()
    model, history = train_supervised(
        tasks,
        config=TrainingConfig(
            epochs=5,
            learning_rate=1e-3,
            max_samples_per_task=8,
            checkpoint_path="/tmp/alphareason_demo.pt",
        ),
    )
    print("Trained on", len(tasks), "tasks.")
    print("Final epoch:", history[-1])
    print("Checkpoint:", "/tmp/alphareason_demo.pt")
