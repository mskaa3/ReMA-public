import json
import sys
from pathlib import Path

try:
    from verl.hierarchical_rema import train as train_module
except ModuleNotFoundError:
    from hierarchical_rema import train as train_module


def _write_rollout(path: Path, task_id: str = "task-1") -> None:
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


def test_train_module_imports() -> None:
    assert hasattr(train_module, "main")


def test_train_export_only_writes_policy_artifacts(tmp_path, monkeypatch) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_rollout(rollout_dir / "all_rollouts.jsonl", task_id="task-42")

    output_dir = tmp_path / "trained"
    monkeypatch.setattr(
        train_module,
        "run_offline_policy_training",
        lambda train_samples, val_samples, config: {
            "output_dir": str(Path(config.output_dir)),
            "steps": 1,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "objective": "grpo",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--input",
            str(rollout_dir),
            "--output-dir",
            str(output_dir),
            "--save-replay-copy",
        ],
    )

    train_module.main()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))

    assert sorted(manifest["policies"].keys()) == ["decomposer_controller", "selector_controller"]
    assert sorted(summary.keys()) == ["decomposer_controller", "selector_controller"]

    for policy_id in summary:
        policy_dir = output_dir / policy_id
        assert (policy_dir / "train_samples.jsonl").exists()
        assert (policy_dir / "all_samples.jsonl").exists()
