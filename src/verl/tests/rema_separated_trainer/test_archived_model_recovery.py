"""Exercise archive discovery and recovery without Slurm, S3, or model weights."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/recover_archived_model.sh"


class ArchivedModelRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "finished_jobs" / "6099558"
        self.archive.mkdir(parents=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, ARCHIVE_ROOT=str(self.archive.parent),
                        RECOVERY_REPO=str(ROOT), RECOVERY_SCRATCH=str(self.root / "scratch"),
                        PATH=f"{self.bin}:{os.environ['PATH']}",
                        CALL_LOG=str(self.root / "calls"))
        self.env.pop("SLURM_JOB_ID", None)

    def run_script(self, *args, answers=""):
        return subprocess.run(["bash", str(SCRIPT), *args], env=self.env,
                              input=answers, capture_output=True, text=True, timeout=10)

    def actor(self, node="r14-10", step=50, role="worker_stage_4"):
        actor = (self.archive / "LOCAL" / node / "agent12" / "checkpoints"
                 / "agent12" / f"global_step_{step}" / role / "actor")
        actor.mkdir(parents=True)
        return actor

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    def test_plain_archive_lists_all_nodes_without_allocation(self):
        left, right = self.actor(), self.actor(node="r14-5")
        result = self.run_script("--list", "6099558")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(left), result.stdout)
        self.assertIn(str(right), result.stdout)
        self.assertIn("global_step_50/worker_stage_4/actor", result.stdout)
        self.assertFalse((self.root / "scratch").exists())

    def test_linked_node_is_discovered(self):
        original = self.actor()
        node = self.archive / "LOCAL" / "r14-10"
        target = self.root / "archived-node"
        node.rename(target)
        node.symlink_to(target, target_is_directory=True)
        result = self.run_script("--list", "6099558")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(original), result.stdout)

    def test_linked_job_and_actor_are_discovered(self):
        actor = self.actor()
        actor_target = self.root / "actor-data"
        actor.rename(actor_target)
        actor.symlink_to(actor_target, target_is_directory=True)
        job_target = self.root / "job-data"
        self.archive.rename(job_target)
        self.archive.symlink_to(job_target, target_is_directory=True)
        result = self.run_script("--list", "6099558")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(actor), result.stdout)

    def test_empty_archive_fails_before_reading_answers(self):
        result = self.run_script("6099558", answers="50\nworker_stage_4\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No checkpoint actor directories found", result.stderr)
        diagnostic = next(line.strip() for line in result.stderr.splitlines()
                          if line.startswith("  find "))
        self.assertIn("model_world_size_*_rank_*.pt", shlex.split(diagnostic))
        self.assertNotIn("Available checkpoints", result.stdout)
        self.assertFalse((self.root / "scratch").exists())

    def test_broken_link_does_not_claim_checkpoint_is_available(self):
        local = self.archive / "LOCAL"
        local.mkdir()
        (local / "r14-10").symlink_to(self.root / "missing-node", target_is_directory=True)
        result = self.run_script("--list", "6099558")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("broken links", result.stderr)

    def test_list_continues_to_second_job_after_empty_archive(self):
        actor = self.actor()
        self.archive.rename(self.archive.parent / "6099557")
        self.archive.mkdir()
        result = self.run_script("--list", "6099558", "6099557")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No checkpoint actor directories found", result.stderr)
        self.assertIn(str(actor).replace("6099558", "6099557"), result.stdout)

    def test_search_error_does_not_use_partial_results(self):
        actor = self.actor()
        self.stub("find", f'echo "{actor}"\necho "Permission denied" >&2\nexit 1\n')
        result = self.run_script("--list", "6099558")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Permission denied", result.stderr)
        self.assertIn("Checkpoint search failed", result.stderr)
        self.assertNotIn("Available checkpoints", result.stdout)

    def test_wrong_step_reports_selection_error(self):
        self.actor(step=100)
        result = self.run_script("6099558", answers="50\nworker_stage_4\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No checkpoint for step 50, role worker_stage_4", result.stderr)
        self.assertIn("global_step_100/worker_stage_4/actor", result.stdout)

    def prepare_recovery(self):
        self.env["SLURM_JOB_ID"] = "9999999"
        left, right = self.actor(), self.actor(node="r14-5")
        (left / "model_world_size_2_rank_0.pt").write_bytes(b"rank0")
        (right / "model_world_size_2_rank_1.pt").write_bytes(b"rank1")
        hf = left / "huggingface"
        hf.mkdir()
        (hf / "config.json").write_text('{"model_type":"qwen2"}')
        image = self.root / "image.sif"
        image.write_bytes(b"fake-container")
        (self.archive / "verl-rema-v3.sif").symlink_to(image)
        self.stub("apptainer", '''
printf 'apptainer %s\n' "$*" >> "$CALL_LOG"
while (($#)); do
    if [[ "$1" == --local_dir ]]; then
        printf 'merged' > "$2/huggingface/model.safetensors"
        exit 0
    fi
    shift
done
''')
        self.stub("rclone", 'printf "rclone %s\\n" "$*" >> "$CALL_LOG"\n')
        return left, right

    def test_recovery_collects_all_ranks_and_uploads_one_worker_model(self):
        self.prepare_recovery()
        result = self.run_script("6099558", answers="50\nworker_stage_4\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = (self.root / "calls").read_text()
        self.assertIn("model_merger.py", calls)
        self.assertIn("validate_hf_model_dir.py", calls)
        self.assertIn("/global_step_50/workers/huggingface", calls)
        self.assertIn("rclone check", calls)
        self.assertIn("Merging 2 ranks", result.stdout)

    def test_incomplete_checkpoint_is_not_merged_or_uploaded(self):
        _, right = self.prepare_recovery()
        (right / "model_world_size_2_rank_1.pt").unlink()
        result = self.run_script("6099558", answers="50\nworker_stage_4\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Incomplete/mixed checkpoint", result.stderr)
        self.assertFalse((self.root / "calls").exists())

    def test_conflicting_shards_are_not_merged_or_uploaded(self):
        _, right = self.prepare_recovery()
        (right / "model_world_size_2_rank_0.pt").write_bytes(b"different-rank0")
        result = self.run_script("6099558", answers="50\nworker_stage_4\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Conflicting duplicate shard", result.stderr)
        self.assertFalse((self.root / "calls").exists())


if __name__ == "__main__":
    unittest.main()
