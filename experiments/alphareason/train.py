from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from .dataset import SupervisedSample, build_supervised_dataset
from .policies import AlphaReasonPolicyValueNet, action_to_key
from .run_experiment import CHALLENGE100_PATH, build_demo_sudoku_task
from .task import AlphaReasonTask
from .tasks.sudoku import build_standard_sudoku_task
from demonstrations.sudoku.sudoku_bench.read_puzzle import initial_board_into_evidence

LOCAL_SUDOKU_DATASET_ROOT = Path(__file__).resolve().parents[2] / "demonstrations" / "sudoku" / "SudokuDataset"
HF_SUDOKU_DATASET_CACHE_ROOT = Path(
    "/home/schuette/.cache/huggingface/hub/datasets--Ritvik19--Sudoku-Dataset"
)
HF_SUDOKU_DATASET_BLOB_MAP = {
    "train_0.parquet": "c67780404ff7b82a4f9fc84bbff3aa20fecba47e6d3356374e03d58d090fffdb",
    "train_1.parquet": "7e7d8bd405e85397e7da838566600eacca7b00215ffd7431d9c4d78db81c0c59",
    "train_0_aux.parquet": "8fed3620a0d9dd81a969a36a450f5e5667631eb60d1be777e5cb08053df88a4d",
    "valid_0.parquet": "4fcfcde6c5d86bf1b936d2326f4021de83a3db17c58112038213d3e32e8b8b24",
    "valid_1.parquet": "0f0a3d7bf6876fc5726845e93edb01edf151a84aea527e945434fe4ea611d9a5",
    "README.md": "1395ae06a71c498da2faed4040b93e0cfd02124e",
    ".gitattributes": "28df5f900b358436f0267334b3e3e9af33f917ba",
}


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_samples_per_task: Optional[int] = None
    shuffle_good_actions: bool = True
    seed: int = 0
    checkpoint_path: Optional[str] = None


def _candidate_sudoku_dataset_dirs() -> List[Path]:
    candidates: List[Path] = []
    if LOCAL_SUDOKU_DATASET_ROOT.exists():
        for path in sorted(LOCAL_SUDOKU_DATASET_ROOT.iterdir()):
            if path.is_dir():
                candidates.append(path)
    snapshots_root = HF_SUDOKU_DATASET_CACHE_ROOT / "snapshots"
    if snapshots_root.exists():
        for path in sorted(snapshots_root.iterdir()):
            if path.is_dir():
                candidates.append(path)
    return candidates


def _resolve_dataset_file(dataset_dir: Path, file_name: str) -> Path:
    candidate = dataset_dir / file_name
    if candidate.exists():
        return candidate
    if candidate.is_symlink():
        target = candidate.readlink()
        resolved = (candidate.parent / target).resolve(strict=False)
        if resolved.exists():
            return resolved
        blob_match = re.search(r"blobs/([^/]+)$", str(target))
        if blob_match:
            blob_path = HF_SUDOKU_DATASET_CACHE_ROOT / "blobs" / blob_match.group(1)
            if blob_path.exists():
                return blob_path
    raise FileNotFoundError(f"Could not resolve dataset file {candidate}.")


def _find_sudoku_dataset_dir(split: str) -> Optional[Path]:
    pattern = re.compile(rf"{re.escape(split)}_\d+\.parquet$")
    for dataset_dir in _candidate_sudoku_dataset_dirs():
        try:
            filenames = [path.name for path in dataset_dir.iterdir()]
        except FileNotFoundError:
            continue
        if any(pattern.match(name) for name in filenames):
            return dataset_dir
    return None


def _hf_blob_split_files(split: str) -> List[tuple[int, Path]]:
    pattern = re.compile(rf"{re.escape(split)}_(\d+)\.parquet$")
    blobs_root = HF_SUDOKU_DATASET_CACHE_ROOT / "blobs"
    shard_files = []
    for file_name, blob_name in HF_SUDOKU_DATASET_BLOB_MAP.items():
        match = pattern.fullmatch(file_name)
        if not match:
            continue
        blob_path = blobs_root / blob_name
        if blob_path.exists():
            shard_files.append((int(match.group(1)), blob_path))
    return sorted(shard_files, key=lambda item: item[0])


def load_hf_sudoku_dataset_training_tasks(
    limit: Optional[int] = None,
    offset: int = 0,
    split: str = "train",
    batch_size: int = 256,
    num: int = 3,
    dataset_dir: Optional[str] = None,
) -> List[AlphaReasonTask]:
    if offset < 0:
        raise ValueError("offset must be non-negative.")

    dataset_path = Path(dataset_dir) if dataset_dir is not None else _find_sudoku_dataset_dir(split=split)
    shard_files: List[tuple[int, Path]] = []
    if dataset_path is not None:
        pattern = re.compile(rf"{re.escape(split)}_(\d+)\.parquet$")
        for path in sorted(dataset_path.iterdir(), key=lambda item: item.name):
            match = pattern.fullmatch(path.name)
            if match:
                shard_files.append((int(match.group(1)), _resolve_dataset_file(dataset_path, path.name)))
    else:
        shard_files = _hf_blob_split_files(split=split)
    if not shard_files:
        raise FileNotFoundError("Could not find readable parquet shards for the SudokuDataset.")

    tasks: List[AlphaReasonTask] = []
    seen = 0
    for shard_idx, parquet_path in shard_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["puzzle", "solution"]):
            batch_dict = batch.to_pydict()
            puzzles = batch_dict["puzzle"]
            solutions = batch_dict["solution"]
            for row_idx, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
                if seen < offset:
                    seen += 1
                    continue
                initial_evidence = initial_board_into_evidence(puzzle, colNum=num, rowNum=num)
                solution_evidence = initial_board_into_evidence(solution, colNum=num, rowNum=num)
                task = build_standard_sudoku_task(
                    initial_evidence=initial_evidence,
                    solution_evidence=solution_evidence,
                    num=num,
                    name=f"sudoku_dataset_{split}_{seen}",
                )
                task.metadata.update(
                    {
                        "source": "hf_sudoku_dataset",
                        "split": split,
                        "row_index": seen,
                        "shard_file": parquet_path.name,
                        "shard_index": shard_idx,
                    }
                )
                tasks.append(task)
                seen += 1
                if limit is not None and len(tasks) >= limit:
                    return tasks
    return tasks


def load_challenge100_training_tasks(
    limit: Optional[int] = None,
    offset: int = 0,
    num: int = 3,
    csv_path: str = CHALLENGE100_PATH,
) -> List[AlphaReasonTask]:
    df = pd.read_csv(csv_path)
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if limit is None:
        selected = df.iloc[offset:]
    else:
        selected = df.iloc[offset : offset + limit]

    tasks: List[AlphaReasonTask] = []
    for row_idx, row in selected.iterrows():
        initial_evidence = initial_board_into_evidence(row["initial_board"], colNum=num, rowNum=num)
        solution_evidence = initial_board_into_evidence(row["solution"], colNum=num, rowNum=num)
        puzzle_id = row.get("puzzle_id", row_idx)
        task = build_standard_sudoku_task(
            initial_evidence=initial_evidence,
            solution_evidence=solution_evidence,
            num=num,
            name=f"challenge100_{puzzle_id}",
        )
        task.metadata.update(
            {
                "source": "challenge100",
                "row_index": int(row_idx),
                "puzzle_id": puzzle_id,
                "title": row.get("title"),
                "author": row.get("author"),
            }
        )
        tasks.append(task)
    return tasks


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


def build_default_training_tasks(
    use_hf_sudoku_dataset: bool = True,
    hf_sudoku_limit: int = 8,
    hf_sudoku_offset: int = 0,
    use_challenge100: bool = True,
    challenge100_limit: int = 8,
    challenge100_offset: int = 0,
) -> List[AlphaReasonTask]:
    if use_hf_sudoku_dataset:
        try:
            tasks = load_hf_sudoku_dataset_training_tasks(
                limit=hf_sudoku_limit,
                offset=hf_sudoku_offset,
                split="train",
            )
        except FileNotFoundError:
            tasks = []
        if tasks:
            return tasks
    if use_challenge100:
        tasks = load_challenge100_training_tasks(
            limit=challenge100_limit,
            offset=challenge100_offset,
        )
        if tasks:
            return tasks
    return build_demo_training_tasks()


if __name__ == "__main__":
    tasks = build_default_training_tasks(
        use_hf_sudoku_dataset=True,
        hf_sudoku_limit=8,
        use_challenge100=True,
        challenge100_limit=8,
    )
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
    print("First task:", tasks[0].name)
    print("Final epoch:", history[-1])
    print("Checkpoint:", "/tmp/alphareason_demo.pt")
