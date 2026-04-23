import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .actions import AddClusterAction, AssignValueAction
from .data_loaders import (
    REPO_CHALLENGE100_CSV_PATH,
    REPO_HF_SUDOKU_DATASET_DIR,
    load_challenge100_training_tasks,
    load_hf_sudoku_dataset_training_tasks,
    load_prepared_sudoku_training_tasks,
)
from .dataset import build_supervised_dataset
from .policies import AlphaReasonPolicyValueNet, action_to_key
from .task import AlphaReasonTask


def infer_model_shape(tasks: Sequence[AlphaReasonTask]) -> tuple[int, int]:
    num_features = max(len(task.target_feature_keys) for task in tasks)
    max_feature_dim = max(max(task.feature_domain_sizes.values()) for task in tasks)
    return num_features, max_feature_dim


def _encode_state_cpu(state, *, num_features: int, max_feature_dim: int) -> torch.Tensor:
    assigned = torch.zeros((num_features, max_feature_dim), dtype=torch.float32, device="cpu")
    priors = torch.zeros((num_features, max_feature_dim), dtype=torch.float32, device="cpu")
    assigned_mask = torch.zeros(num_features, dtype=torch.float32, device="cpu")
    unresolved_mask = torch.zeros(num_features, dtype=torch.float32, device="cpu")

    feature_index = {feature_key: idx for idx, feature_key in enumerate(state.target_feature_keys[:num_features])}
    for feature_key, idx in feature_index.items():
        domain_size = min(state.feature_domain_sizes[feature_key], max_feature_dim)
        feature_probs = state.feature_priors.get(feature_key, tuple())
        for value_index in range(min(len(feature_probs), domain_size)):
            priors[idx, value_index] = float(feature_probs[value_index])
        if feature_key in state.target_assignments:
            value_index = state.target_assignments[feature_key]
            if value_index < max_feature_dim:
                assigned[idx, value_index] = 1.0
            assigned_mask[idx] = 1.0
        else:
            unresolved_mask[idx] = 1.0

    summary = torch.tensor(
        [
            float(state.assigned_count) / max(1.0, float(len(state.target_feature_keys))),
            float(len(state.active_inference_clusters)) / 32.0,
            float(len(state.cluster_candidates)) / 32.0,
            float(state.value_estimate),
        ],
        dtype=torch.float32,
        device="cpu",
    )
    return torch.cat(
        [
            assigned.flatten(),
            priors.flatten(),
            assigned_mask,
            unresolved_mask,
            summary,
        ],
        dim=0,
    )


def _encode_actions_cpu(
    state,
    legal_actions: List[object],
    *,
    num_features: int,
    max_feature_dim: int,
) -> torch.Tensor:
    action_dim = num_features + max_feature_dim + 3
    if not legal_actions:
        return torch.zeros((0, action_dim), dtype=torch.float32, device="cpu")

    feature_index = {feature_key: idx for idx, feature_key in enumerate(state.target_feature_keys[:num_features])}
    action_vectors: List[torch.Tensor] = []
    for action in legal_actions:
        vector = torch.zeros(action_dim, dtype=torch.float32, device="cpu")
        if isinstance(action, AssignValueAction):
            if action.feature_key in feature_index:
                vector[feature_index[action.feature_key]] = 1.0
            if action.value_index < max_feature_dim:
                vector[num_features + action.value_index] = 1.0
            vector[num_features + max_feature_dim] = 1.0
        elif isinstance(action, AddClusterAction):
            vector[num_features + max_feature_dim + 1] = 1.0
            vector[num_features + max_feature_dim + 2] = min(1.0, len(action.cluster_key) / 128.0)
        action_vectors.append(vector)
    return torch.stack(action_vectors, dim=0)


class _EncodedSampleDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        return self._rows[idx]


def _collate_encoded_batch(rows: List[dict]):
    batch_size = len(rows)
    max_actions = max(row["action_tensor"].shape[0] for row in rows) if rows else 0

    state = torch.stack([row["state_tensor"] for row in rows], dim=0)
    value_target = torch.tensor([row["value_target"] for row in rows], dtype=torch.float32)

    action_dim = rows[0]["action_dim"]
    actions = torch.zeros((batch_size, max_actions, action_dim), dtype=torch.float32)
    target_probs = torch.zeros((batch_size, max_actions), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)

    for i, row in enumerate(rows):
        a = row["action_tensor"].shape[0]
        if a:
            actions[i, :a, :] = row["action_tensor"]
            target_probs[i, :a] = row["target_probs"]
            mask[i, :a] = True

    return {
        "state": state,
        "actions": actions,
        "target_probs": target_probs,
        "mask": mask,
        "value_target": value_target,
    }


def _maybe_init_ddp(device: torch.device) -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0
    if device.type != "cuda":
        raise ValueError("DDP requested via WORLD_SIZE>1 but device is not CUDA.")
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank


def train_supervised(
    tasks: Sequence[AlphaReasonTask],
    model: Optional[AlphaReasonPolicyValueNet] = None,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    shuffle_good_actions: bool = True,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    log_every_epoch: bool = False,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
    use_amp: bool = True,
) -> tuple[AlphaReasonPolicyValueNet, List[Dict[str, float]]]:
    if not tasks:
        raise ValueError("train_supervised requires at least one task.")

    device = torch.device(device)
    ddp_enabled, ddp_rank, ddp_local_rank = _maybe_init_ddp(device)
    if ddp_enabled:
        seed = int(seed) + ddp_rank
        device = torch.device("cuda", ddp_local_rank)

    num_features, max_feature_dim = infer_model_shape(tasks)
    model = model or AlphaReasonPolicyValueNet(
        num_features=num_features,
        max_feature_dim=max_feature_dim,
        device=device,
    )
    if model.device != device:
        model.device = device
        model.to(device)
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    supervised_samples = build_supervised_dataset(tasks, shuffle_good_actions=shuffle_good_actions, seed=seed)
    if not supervised_samples:
        raise ValueError("No supervised samples could be extracted from the provided tasks.")

    action_dim = num_features + max_feature_dim + 3
    encoded_rows: List[dict] = []
    for sample in supervised_samples:
        legal_actions = list(sample.state.legal_actions)
        state_tensor = _encode_state_cpu(sample.state, num_features=num_features, max_feature_dim=max_feature_dim)
        action_tensor = _encode_actions_cpu(
            sample.state, legal_actions, num_features=num_features, max_feature_dim=max_feature_dim
        )

        target_probs = torch.zeros(len(legal_actions), dtype=torch.float32, device="cpu")
        action_index = {action_to_key(action): idx for idx, action in enumerate(legal_actions)}
        for action, prob in sample.policy_target.items():
            key = action_to_key(action)
            if key in action_index:
                target_probs[action_index[key]] = float(prob)
        total = float(target_probs.sum().item())
        if total > 0.0:
            target_probs = target_probs / total

        encoded_rows.append(
            {
                "state_tensor": state_tensor,
                "action_tensor": action_tensor,
                "target_probs": target_probs,
                "value_target": float(sample.value_target),
                "action_dim": action_dim,
            }
        )

    dataset = _EncodedSampleDataset(encoded_rows)
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=False) if ddp_enabled else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        collate_fn=_collate_encoded_batch,
        pin_memory=(device.type == "cuda"),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        step_count = 0

        model.train()
        for batch in loader:
            state = batch["state"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            target_probs = batch["target_probs"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            value_target = batch["value_target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                raw_model = model.module if hasattr(model, "module") else model
                state_hidden = raw_model.trunk(state)
                value = raw_model.value_head(state_hidden).squeeze(-1)

                expanded_hidden = state_hidden.unsqueeze(1).expand(actions.shape[0], actions.shape[1], -1)
                flat_inp = torch.cat([expanded_hidden, actions], dim=2).reshape(
                    -1, expanded_hidden.shape[2] + actions.shape[2]
                )
                flat_scores = raw_model.action_head(flat_inp).reshape(actions.shape[0], actions.shape[1]).squeeze(-1)

                logits = flat_scores.float().masked_fill(~mask, -1e9)
                log_probs = F.log_softmax(logits, dim=1)

                target_sum = target_probs.sum(dim=1, keepdim=True)
                legal_counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=torch.float32)
                uniform = mask.to(dtype=torch.float32) / legal_counts
                normed_targets = torch.where(
                    target_sum > 0.0, target_probs / target_sum.clamp(min=1e-12), uniform
                )

                policy_loss = -(normed_targets * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(value, value_target)
                total_loss = policy_loss + value_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += float(total_loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            step_count += 1

        if ddp_enabled:
            sums = torch.tensor(
                [total_loss_sum, policy_loss_sum, value_loss_sum, float(step_count)],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            total_loss_sum, policy_loss_sum, value_loss_sum, step_count = map(float, sums.tolist())

        epoch_stats = {
            "epoch": float(epoch + 1),
            "total_loss": total_loss_sum / max(1.0, step_count),
            "policy_loss": policy_loss_sum / max(1.0, step_count),
            "value_loss": value_loss_sum / max(1.0, step_count),
            "samples": float(len(dataset)),
        }
        if (not ddp_enabled) or ddp_rank == 0:
            history.append(epoch_stats)
        if log_every_epoch and ((not ddp_enabled) or ddp_rank == 0):
            print(f"Epoch {epoch + 1}/{epochs}: total_loss={epoch_stats['total_loss']:.6f}", flush=True)

    if checkpoint_path and ((not ddp_enabled) or ddp_rank == 0):
        checkpoint_file = Path(checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "num_features": raw_model.num_features,
                "max_feature_dim": raw_model.max_feature_dim,
                "history": history,
            },
            checkpoint_file,
        )

    return model, history


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entrypoint.")
    prepared_parquet = "/tmp/alphareason_prepared_unsolved.parquet"
    tasks = load_prepared_sudoku_training_tasks(parquet_path=prepared_parquet, num_samples=None, num=3)
    model, history = train_supervised(
        tasks,
        epochs=1,
        learning_rate=1e-3,
        checkpoint_path="/tmp/alphareason_demo.pt",
        log_every_epoch=True,
        device="cuda",
        batch_size=512,
        use_amp=True,
    )
    print("Trained on", len(tasks), "tasks.")
    print("First task:", tasks[0].name)
    print("Final epoch:", history[-1])
    print("Checkpoint:", "/tmp/alphareason_demo.pt")
