"""Keep checkpoint upload allowlists ordered and exclude optimizer state."""

from pathlib import Path
import shlex
import subprocess

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("launcher", [
    "agent0-rl-trainer-multinode.sh",
    "agent12-curriculum-trainer-multinode.sh",
])
def test_checkpoint_export_uses_ordered_filters(launcher):
    path = ROOT / launcher
    subprocess.run(["bash", "-n", str(path)], check=True)
    commands = [
        shlex.split(line)
        for line in path.read_text().replace("\\\n", "").splitlines()
        if line.lstrip().startswith('rclone copy "$actor_dir" ')
    ]
    assert len(commands) == 1
    args = commands[0]
    assert "--include" not in args
    assert "--exclude" not in args
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "--filter"] == [
        "+ /model_world_size_*_rank_*.pt",
        "+ /huggingface/**",
        "- **",
    ]


@pytest.mark.parametrize("config_name", ["agent0-rl.yaml", "rema-rl.yaml"])
def test_exportable_checkpoints_include_hf_metadata(config_name):
    config = OmegaConf.load(ROOT / "config" / config_name)
    contents = list(config.actor_rollout_ref.actor.checkpoint.contents)
    assert {"model", "hf_model", "optimizer", "extra"} <= set(contents)
