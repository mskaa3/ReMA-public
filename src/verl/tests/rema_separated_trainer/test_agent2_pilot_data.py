import importlib.util
from pathlib import Path

import pandas as pd
import pytest


spec = importlib.util.spec_from_file_location(
    "prepare_agent2_pilot_data",
    Path(__file__).resolve().parents[4] / "scripts/prepare_agent2_pilot_data.py",
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def _frame(size=10):
    return pd.DataFrame({
        "question": [f"question {i}" for i in range(size)],
        "teacher_attempts": [["", f"attempt {i}", "another attempt"] for i in range(size)],
        "reward_model": [{"ground_truth": str(i)} for i in range(size)],
        "data_source": ["math"] * size,
    })


def test_pilot_split_is_disjoint_reproducible_and_keeps_all_training_attempts(tmp_path):
    source = tmp_path / "source.parquet"
    frame = _frame()
    frame.to_parquet(source, index=False)
    for folder in ("first", "second"):
        pilot.prepare_pilot(source, tmp_path / folder, train_questions=6, val_questions=3)
    train = pd.read_parquet(tmp_path / "first/train.parquet")
    val = pd.read_parquet(tmp_path / "first/val.parquet")
    assert len(train) == 6 and len(val) == 3
    assert not set(train.question) & set(val.question)
    assert all(len(attempts) == 3 for attempts in train.teacher_attempts)
    assert all(value.startswith("attempt ") for value in val.teacher_attempt)
    assert "teacher_attempts" not in val.columns
    assert val["subset"].tolist() == ["math"] * 3
    pd.testing.assert_frame_equal(val, pd.read_parquet(tmp_path / "second/val.parquet"))
    pd.testing.assert_frame_equal(train, pd.read_parquet(tmp_path / "second/train.parquet"))
    assert (tmp_path / "first/split.json").exists()


@pytest.mark.parametrize("problem,message", [
    ("duplicate", "duplicate questions"),
    ("small", "too small"),
    ("no_attempts", "non-empty teacher attempts"),
])
def test_pilot_rejects_unsafe_or_incomplete_inputs(tmp_path, problem, message):
    frame = _frame(2 if problem == "small" else 10)
    if problem == "duplicate":
        frame.loc[1, "question"] = "  question   0 "
    if problem == "no_attempts":
        frame["teacher_attempts"] = [[""]] * len(frame)
    source = tmp_path / "source.parquet"
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match=message):
        pilot.prepare_pilot(source, tmp_path / "output", train_questions=6, val_questions=3)
