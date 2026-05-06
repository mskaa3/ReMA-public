import json
import sys
from pathlib import Path

try:
    from verl.hierarchical_rema import train as train_module
except ModuleNotFoundError:
    from hierarchical_rema import train as train_module

def test_train_module_imports() -> None:
    assert hasattr(train_module, "main")


def test_train_load_tasks_from_directory_auto_selects_supported_file(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "all_test_data.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "idx": 7,
                "dataset": "toy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = train_module.load_tasks(
        task_source=str(dataset_dir),
        task_format="auto",
        prompt_key="question",
        answer_key="answer",
        task_id_key="idx",
        max_tasks=0,
    )

    assert len(tasks) == 1
    assert tasks[0].task_id == "7"
    assert tasks[0].prompt == "What is 2 + 2?"
    assert tasks[0].ground_truth == "4"


def test_integrated_train_writes_epoch_artifacts(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "trained"
    monkeypatch.setattr(
        train_module,
        "run_offline_policy_training",
        lambda train_samples, val_samples, config, tracking=None, tracking_prefix="", log_step_offset=0: {
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
            "--task-source",
            "demo",
            "--backend",
            "mock",
            "--num-epochs",
            "1",
            "--disable-rollout-logging",
            "--output-dir",
            str(output_dir),
            "--save-replay-copy",
        ],
    )

    train_module.main()

    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    epoch_dir = output_dir / "epoch_0001"
    manifest = json.loads((epoch_dir / "train" / "manifest.json").read_text(encoding="utf-8"))

    assert summary["num_epochs"] == 1
    assert len(summary["epochs"]) == 1
    assert (epoch_dir / "rollout_summary.json").exists()
    assert (epoch_dir / "epoch_summary.json").exists()
    assert sorted(manifest["policies"].keys()) == ["decomposer_controller", "selector_controller"]

    for policy_id in manifest["policies"]:
        policy_dir = epoch_dir / "train" / policy_id
        assert (policy_dir / "train_samples.jsonl").exists()
        assert (policy_dir / "all_samples.jsonl").exists()
