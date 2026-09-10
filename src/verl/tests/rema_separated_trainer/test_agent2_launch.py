"""CPU launch checks using the real shell overrides and rollout coordinator."""

import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
from verl.rema_separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout
from verl.rema_separated_trainer.ppo.prefix_probe import collect_prefix_probe_requests
from verl.rema_separated_trainer.ppo.ray_trainer import RayReMASeparatedTrainer


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def launch_trainer(tmp_path, request):
    wrapper = (ROOT / "agent2-only-rl-trainer-multinode.sh").read_text()
    launcher = (ROOT / "agent12-curriculum-trainer-multinode.sh").read_text()
    wrapper_exports = wrapper[wrapper.index("export TRAIN_DECOMPOSER="):wrapper.index('exec bash')]
    launcher_exports = launcher[:launcher.index("\nif (( GPUS_PER_NODE")]
    launcher_exports = launcher_exports.replace("source ./env.sh", ": # No cluster credentials in tests")
    training_function = launcher[
        launcher.index("run_agent12_training() {"):
        launcher.index('\nif [[ "$ONLINE_TEACHER_GENERATION"')
    ]
    pilot_setup_start = launcher.rindex('\nif [[ "$AGENT2_PILOT"')
    pilot_setup = launcher[pilot_setup_start:launcher.index('\nstart_ray', pilot_setup_start)]
    capture = tmp_path / "training-command.txt"
    # Exercise Bash expansion of the actual launcher; never execute srun,
    # Apptainer, teacher generation, rclone, or the trainer process.
    shell = "\n".join([
        "set -euo pipefail", wrapper_exports, launcher_exports,
        'srun() { printf "%s\\n" "${!#}" > "$CAPTURE"; }',
        'HEAD_NODE=head; IP_HEAD=head:6379; COMMON_MOUNTS=(--nv)',
        training_function, pilot_setup,
        'run_agent12_training /tmp/test.train.parquet 48 True False',
    ])
    subprocess.run(["bash", "-c", shell], check=True, env={
        "PATH": os.environ["PATH"], "SLURM_JOB_ID": "123",
        "SLURM_NNODES": "6", "CAPTURE": str(capture),
        "AGENT2_PILOT": "true" if getattr(request, "param", False) else "false",
    }, capture_output=True, text=True)
    command = shlex.split(capture.read_text().rsplit(";", 1)[-1])
    assert command[:3] == ["python3", "-m", "verl.rema_separated_trainer.main_ppo"]
    overrides = [arg for arg in command[3:] if not arg.startswith("--")]
    config = OmegaConf.load(ROOT / "config/rema-rl.yaml")
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer.config = config
    trainer.use_critic = False
    trainer.use_rm = False
    trainer._validate_config()
    trainer._init_scoped_c3_grpo()
    trainer._init_prefix_probe()
    return trainer


def test_agent2_actual_launcher_configuration(launch_trainer):
    trainer = launch_trainer
    cfg = trainer.config
    hierarchy = trainer._get_hierarchy_config()
    assert cfg.algorithm.switch_agent.model_paths == [
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
    ]
    assert not hierarchy["agent12_curriculum"]["train_decomposer"]
    assert trainer._get_train_agent_roles() == hierarchy["stage_roles"]
    assert len(hierarchy["stage_roles"]) == hierarchy["max_planned_subtasks"] == 4
    assert hierarchy["terminal_worker_as_answer"]
    assert hierarchy["routing_mode"] == "sequential_plan"
    assert trainer.scoped_c3_grpo_enabled and trainer.prefix_probe_enabled
    assert cfg.actor_rollout_ref.rollout.max_num_turns == 1
    assert cfg.actor_rollout_ref.rollout.n == 16
    assert cfg.trainer.nnodes == 6 and cfg.trainer.n_gpus_per_node == 2
    assert cfg.data.train_batch_size * 16 * cfg.actor_rollout_ref.rollout.n == 768
    assert cfg.actor_rollout_ref.rollout.prompt_length + cfg.data.max_response_length <= cfg.actor_rollout_ref.rollout.max_model_len
    assert cfg.algorithm.filter_groups.skip_update_on_zero_trainable
    assert cfg.algorithm.filter_groups.allow_sub_minibatch_on_exhaustion
    assert cfg.trainer.val_before_train
    assert cfg.trainer.test_freq == 10
    assert cfg.trainer.wandb_run_id == "agent2-only-123"


@pytest.mark.parametrize("task_count,focal_index,validation", [
    (1, 1, False), (2, 1, False), (3, 2, False), (4, 4, False),
    (2, 3, False), (3, 2, True), (3, 2, "teacher_assisted"),
])
def test_routed_rollout_visibility_and_c3(launch_trainer, task_count, focal_index, validation):
    trainer = launch_trainer
    hierarchy = trainer._get_hierarchy_config()
    roles = hierarchy["agent_roles"]
    focal = f"worker_stage_{focal_index}"
    terminal = f"worker_stage_{task_count}"
    size = 4
    question = "ORIGINAL_QUESTION_SENTINEL"
    teacher = "PRIVATE_TEACHER_SENTINEL"
    plan = "REASONING: PRIVATE_PLANNER_REASONING\nPLAN:\n" + "\n".join(
        f"- S{i}: Compute intermediate {i}." for i in range(1, task_count + 1)
    )
    prompts = DataProto.from_dict(
        tensors={"batch_idx": torch.arange(size)},
        non_tensors={
            "question": np.array([question] * size, dtype=object),
            "teacher_attempt": np.array([teacher] * size, dtype=object),
            "uid": np.array(["shared-prefix"] * size, dtype=object),
        },
        meta_info={"c3_focal_role": focal, "teacher_attempt_probability": 1.0,
                   "worker_question_probability": 1.0, "validate": bool(validation),
                   "teacher_assisted_validation": validation == "teacher_assisted"},
    )
    rollout = MultiAgentRollout.__new__(MultiAgentRollout)
    rollout.config = SimpleNamespace(stop_when_truncated=True)
    generated_chats = {}

    def generate(role, chats, tokenizers, meta_info, response_length, max_new_tokens=None):
        generated_chats[role] = chats
        outputs = [plan] * len(chats) if role == "decomposer" else [
            f"REASONING: PRIVATE_{role}_REASONING\nLOCAL_RESULT: \\boxed{{{i + 10}}}"
            for i in range(len(chats))
        ]
        return outputs, [len(text) for text in outputs], ["stop"] * len(outputs), None, [
            list(text.encode()) for text in outputs
        ]

    rollout._generate_from_chat_list = generate
    history, flags, reasons = rollout._initialize_conversation_state(size)
    outputs, conversations, actions, state = rollout._run_hierarchical_conversation(
        prompts, {}, 1, roles, {role: "role instructions" for role in roles},
        hierarchy, history, flags, reasons, 1024, None,
    )
    assert "selector" not in generated_chats
    assert state["terminal_stage_roles"] == [terminal] * size
    assert all("\\boxed{" in answer for answer in outputs)
    assert reasons == ["final_boxed_answer"] * size
    for records in history:
        assert records[0]["planned_subtask_count"] == task_count
        assert [r["role"] for r in records if r["executed"]] == [
            "decomposer", *hierarchy["stage_roles"][:task_count],
        ]
        requests = collect_prefix_probe_requests(
            [records], [terminal], focal_role=focal, decomposer_role="decomposer",
            stage_roles=hierarchy["stage_roles"],
        )
        terminal_input = next(r.message for r in requests if r.source_kind == "terminal")
        assert f"Compute intermediate {task_count}" in terminal_input
        assert "CURRENT TASK:" in terminal_input
        assert "PREVIOUS LOCAL RESULTS:" not in terminal_input
        assert "PRIVATE_" not in terminal_input and question not in terminal_input
        assert "\\boxed{10}" not in terminal_input
        for earlier in range(1, task_count):
            assert f"Compute intermediate {earlier}" not in terminal_input
    for chat in generated_chats["decomposer"]:
        text = "\n".join(message["content"] for message in chat)
        assert question in text
        assert (teacher in text) is (not validation or validation == "teacher_assisted")
    for index, role in enumerate(hierarchy["stage_roles"][:task_count], 1):
        for chat in generated_chats[role]:
            text = "\n".join(message["content"] for message in chat)
            assert (question in text) is (index < task_count)
            assert teacher not in text and "PRIVATE_PLANNER_REASONING" not in text
            assert "Compute intermediate " + str(index) in text
            assert "PRIVATE_worker" not in text
            assert ("PREVIOUS LOCAL RESULTS:" in text) is (index > 1)
        expected_count = size if validation or index >= focal_index else 1
        assert len(generated_chats[role]) == expected_count
    assert len(generated_chats["decomposer"]) == (size if validation else 1)
    if validation or focal_index > task_count:
        assert actions == [None] * size
    else:
        assert all(action["role"] == focal for action in actions)
        assert all(action["chat"] == actions[0]["chat"] for action in actions)
        assert len({action["output"] for action in actions}) == size


@pytest.mark.parametrize("launch_trainer", [True], indirect=True)
def test_pilot_preserves_rollouts_and_shortens_training(launch_trainer):
    cfg = launch_trainer.config
    assert cfg.trainer.total_training_steps == 20
    assert cfg.trainer.test_freq == 5
    assert cfg.actor_rollout_ref.rollout.n == 16
    assert cfg.algorithm.hierarchy.agent12_curriculum.worker_bootstrap_steps == 800
    assert cfg.data.teacher_assisted_validation
    assert cfg.data.val_files.endswith("/agent2_pilot/val.parquet")


def test_paired_validation_keeps_standard_accuracy_namespace():
    trainer = RayReMASeparatedTrainer.__new__(RayReMASeparatedTrainer)
    trainer.config = SimpleNamespace(data={"teacher_assisted_validation": True})
    calls = []

    def validate(*, teacher_assisted):
        calls.append(teacher_assisted)
        return {"val/acc/math": .75 if teacher_assisted else .25,
                "val/leakage/terminal/rate": float(teacher_assisted)}

    trainer._validate_context = validate
    metrics = trainer._validate()
    assert calls == [False, True]
    assert metrics["val/acc/math"] == .25
    assert metrics["val/teacher_assisted/acc/math"] == .75
    assert metrics["val/leakage/terminal/rate"] == 0
    assert metrics["val/teacher_assisted/leakage/terminal/rate"] == 1


def test_real_validation_loop_preserves_context_isolation_and_probe_metrics(launch_trainer, tmp_path):
    trainer = launch_trainer
    trainer.config.data.teacher_assisted_validation = True
    trainer.global_steps = 0
    trainer._current_train_agent = None
    hierarchy = trainer._get_hierarchy_config()
    roles = hierarchy["agent_roles"]
    trainer.actor_rollout_wg = {role: SimpleNamespace(world_size=1) for role in roles}
    trainer._tokenizer_for_role = lambda role: SimpleNamespace(eos_token_id=1, pad_token_id=0)
    trainer._build_rollout_meta_info = lambda turns: {
        "agent_roles": roles, "finish_flag": None, "system_prompts": {}, "hierarchy": hierarchy,
    }
    trainer.val_dataloader = [{
        "question": np.array(["held-out question"], dtype=object),
        "teacher_attempt": np.array(["private attempt"], dtype=object),
        "reward_model": np.array([{"ground_truth": "26", "style": "rule"}], dtype=object),
        "data_source": np.array(["math"], dtype=object),
        "subset": np.array(["math"], dtype=object),
    }]
    contexts = []

    def generate(batch):
        assisted = batch.meta_info["teacher_assisted_validation"]
        contexts.append(assisted)
        assert ("teacher_attempt" in batch.non_tensor_batch) is assisted
        assert "reward_model" not in batch.non_tensor_batch
        records = [{"role": role, "content": "", "executed": False} for role in roles]
        for record in records:
            if record["role"] == "decomposer":
                record.update(content="PLAN:\nS1: compute a\nS2: add 20 to a", executed=True, planned_subtask_count=2)
            elif record["role"] == "worker_stage_1":
                record.update(content=r"LOCAL_RESULT: \boxed{6}", executed=True, assigned_subtasks=["S1"])
            elif record["role"] == "worker_stage_2":
                record.update(content=r"LOCAL_RESULT: \boxed{26}" if assisted else r"LOCAL_RESULT: \boxed{27}",
                              executed=True, assigned_subtasks=["S2"], terminal_probe_input="Add 20 to S1.")
        history = np.empty(1, dtype=object)
        history[0] = records
        return DataProto.from_dict(
            tensors={f"{role}_num_gen_tokens": torch.ones(1, 1) for role in roles},
            non_tensors={
                "history": history,
                "response": np.array([r"\boxed{26}" if assisted else r"\boxed{27}"], dtype=object),
                "terminal_stage_role": np.array(["worker_stage_2"], dtype=object),
                "num_turns": np.array([1], dtype=object),
            },
        )

    class Reward:
        num_examine = 1

        def __call__(self, batch):
            score = torch.tensor([float(batch.non_tensor_batch["response"][0] == r"\boxed{26}")])
            return {"acc": score, **{f"{role}_turn_level_reward": score[:, None] for role in roles}}

        def score_responses(self, sources, responses, *args, **kwargs):
            return [float(response == r"\boxed{26}") for response in responses]

    trainer.multi_turn_generate_sequences = generate
    trainer.val_reward_fn = Reward()
    trainer._generate_prefix_probe_responses = lambda messages: (
        [r"\boxed{UNKNOWN}"] * len(messages), [3] * len(messages), ["stop"] * len(messages),
    )
    logged_tables = []
    trainer._maybe_log_val_generations = lambda **kwargs: logged_tables.append(kwargs)
    metrics = trainer._validate()
    assert contexts == [False, True]
    assert len(logged_tables) == 1
    assert metrics["val/acc/math"] == 0
    assert metrics["val/teacher_assisted/acc/math"] == 1
    assert metrics["val/leakage/gated_accuracy"] == 0
    assert metrics["val/teacher_assisted/leakage/gated_accuracy"] == 1
