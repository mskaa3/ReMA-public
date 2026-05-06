import json

try:
    from verl.hierarchical_rema.controller_data import (
        discover_rollout_files,
        group_samples_by_policy,
        load_controller_samples_from_rollouts,
        train_val_split,
    )
except ModuleNotFoundError:
    from hierarchical_rema.controller_data import (
        discover_rollout_files,
        group_samples_by_policy,
        load_controller_samples_from_rollouts,
        train_val_split,
    )


def _write_rollout(path, task_id="task-1"):
    payload = {
        "timestamp": "2026-05-06T00:00:00Z",
        "task_id": task_id,
        "rollout": {
            "task": {"task_id": task_id},
            "training_batch": {
                "decomposer_samples": [
                    {
                        "role": "decomposer",
                        "policy_id": "decomposer_controller",
                        "group_id": task_id,
                        "prompt_text": "decomposer prompt",
                        "completion_text": "{\"nodes\":[]}",
                        "reward": 0.6,
                        "advantage": 0.4,
                        "metadata": {"model_path": "model-a", "parameter_sharing": False},
                    }
                ],
                "selector_samples": [
                    {
                        "role": "selector",
                        "policy_id": "selector_controller",
                        "group_id": f"{task_id}-decomp-0",
                        "prompt_text": "selector prompt",
                        "completion_text": "{\"assignments\":[]}",
                        "reward": 0.7,
                        "advantage": 0.2,
                        "metadata": {"model_path": "model-b", "parameter_sharing": False},
                    }
                ],
                "frozen_roles": [],
            },
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_discover_rollout_files_from_directory(tmp_path) -> None:
    job_dir = tmp_path / "job-1"
    job_dir.mkdir()
    rollout_file = job_dir / "all_rollouts.jsonl"
    _write_rollout(rollout_file)

    files = discover_rollout_files([str(tmp_path)])
    assert files == [rollout_file.resolve()]


def test_load_controller_samples_and_group_by_policy(tmp_path) -> None:
    rollout_file = tmp_path / "all_rollouts.jsonl"
    _write_rollout(rollout_file, task_id="task-42")

    samples = load_controller_samples_from_rollouts([str(rollout_file)])
    assert len(samples) == 2
    grouped = group_samples_by_policy(samples)
    assert sorted(grouped.keys()) == ["decomposer_controller", "selector_controller"]
    assert grouped["decomposer_controller"][0].task_id == "task-42"


def test_train_val_split_keeps_non_empty_train_when_possible(tmp_path) -> None:
    rollout_file = tmp_path / "all_rollouts.jsonl"
    _write_rollout(rollout_file)
    samples = load_controller_samples_from_rollouts([str(rollout_file)])

    train_samples, val_samples = train_val_split(samples, val_ratio=0.5, seed=123)
    assert len(train_samples) >= 1
    assert len(train_samples) + len(val_samples) == len(samples)
