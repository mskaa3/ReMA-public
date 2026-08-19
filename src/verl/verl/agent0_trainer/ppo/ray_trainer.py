# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ReMA-style GRPO loop for the single-agent serial solver.

The model still emits one ordinary response. The only ReMA-specific behavior
retained here is informative-group collection: homogeneous rollout groups have
zero GRPO advantage, so we replace them with mixed groups before optimization.
"""

from pprint import pprint
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.agent0_trainer.group_filter import (
    classify_rollout_groups,
    indices_for_uids,
    ordered_unique_uids,
    usable_partial_prompt_count,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
)
from verl.utils.tracking import Tracking


class Agent0RayPPOTrainer(RayPPOTrainer):
    """Train one serial solver while collecting only informative GRPO groups."""

    def _validate_config(self):
        super()._validate_config()
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("Agent0RayPPOTrainer currently supports GRPO only")
        if self.config.actor_rollout_ref.rollout.n <= 1:
            raise ValueError("Agent0 group filtering requires rollout.n > 1")
        if self.use_critic:
            raise ValueError("Agent0 GRPO does not use a critic")

    def _select_prompt_groups(self, batch: DataProto, prompt_count: int) -> DataProto:
        ordered_uids = ordered_unique_uids(batch.non_tensor_batch["uid"])
        selected_uids = ordered_uids[:prompt_count]
        selected_indices = indices_for_uids(batch.non_tensor_batch["uid"], selected_uids)
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        expected_trajectories = prompt_count * rollout_n
        if len(selected_indices) != expected_trajectories:
            raise ValueError(
                "Each selected Agent 0 prompt must have exactly rollout.n trajectories: "
                f"expected {expected_trajectories}, found {len(selected_indices)}"
            )
        return self._take_trajectories(batch, selected_indices)

    @staticmethod
    def _take_trajectories(batch: DataProto, indices: list[int]) -> DataProto:
        """Index a DataProto while preserving its batched return type."""

        tensor_indices = torch.as_tensor(indices, dtype=torch.long)
        numpy_indices = np.asarray(indices, dtype=np.int64)
        return DataProto(
            batch=batch.batch[tensor_indices],
            non_tensor_batch={
                key: values[numpy_indices]
                for key, values in batch.non_tensor_batch.items()
            },
            meta_info=batch.meta_info,
        )

    def fit(self):
        """Run GRPO with ReMA-style regeneration of homogeneous groups."""

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.global_steps += 1
        last_val_metrics = None

        filter_config = self.config.algorithm.get("filter_groups", {})
        filter_enabled = bool(filter_config.get("enable", False))
        max_num_gen_batches = int(filter_config.get("max_num_gen_batches", 0))
        allow_partial_batch = bool(filter_config.get("allow_partial_batch", True))
        prompt_batch_size = int(self.config.data.train_batch_size)
        prompt_minibatch_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)

        accumulated_batch = None
        mixed_prompt_count = 0
        num_gen_batches = 0
        all_zero_count = 0
        all_one_count = 0
        homogeneous_other_count = 0
        total_prompt_count = 0
        generated_score_sum = 0.0
        generated_trajectory_count = 0

        for _epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                new_batch = DataProto.from_single_dict(batch_dict)

                if "multi_modal_inputs" in new_batch.non_tensor_batch:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    new_batch.non_tensor_batch["uid"] = np.asarray(
                        [str(uuid.uuid4()) for _ in range(len(new_batch))],
                        dtype=object,
                    )
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        reward_tensor = self.reward_fn(new_batch)
                        new_batch.batch["token_level_scores"] = reward_tensor
                        outcome_scores = reward_tensor.sum(dim=-1)
                        new_batch.batch["acc"] = outcome_scores

                    group_result = classify_rollout_groups(
                        new_batch.non_tensor_batch["uid"],
                        outcome_scores.detach().cpu().tolist(),
                    )
                    num_gen_batches += 1
                    all_zero_count += group_result.all_zero_count
                    all_one_count += group_result.all_one_count
                    homogeneous_other_count += group_result.homogeneous_other_count
                    total_prompt_count += group_result.total_count
                    generated_score_sum += outcome_scores.float().sum().item()
                    generated_trajectory_count += outcome_scores.numel()

                    selected_prompt_count = prompt_batch_size
                    if filter_enabled:
                        kept_indices = indices_for_uids(
                            new_batch.non_tensor_batch["uid"],
                            group_result.mixed_uids,
                        )
                        filtered_batch = (
                            self._take_trajectories(new_batch, kept_indices)
                            if kept_indices
                            else None
                        )
                        mixed_prompt_count += len(group_result.mixed_uids)
                        if filtered_batch is not None:
                            accumulated_batch = (
                                filtered_batch
                                if accumulated_batch is None
                                else DataProto.concat([accumulated_batch, filtered_batch])
                            )

                        if mixed_prompt_count < prompt_batch_size:
                            exhausted = (
                                max_num_gen_batches > 0
                                and num_gen_batches >= max_num_gen_batches
                            )
                            if not exhausted:
                                print(
                                    f"Agent 0 collected {mixed_prompt_count}/{prompt_batch_size} "
                                    f"mixed prompts after {num_gen_batches} generation batches; continuing."
                                )
                                continue

                            selected_prompt_count = usable_partial_prompt_count(
                                mixed_prompt_count,
                                prompt_batch_size,
                                prompt_minibatch_size,
                            )
                            if not allow_partial_batch or selected_prompt_count == 0:
                                print(
                                    "Agent 0 group filter exhausted its generation budget without "
                                    "an optimizer-compatible mixed batch; skipping these rollouts."
                                )
                                accumulated_batch = None
                                mixed_prompt_count = 0
                                num_gen_batches = 0
                                all_zero_count = 0
                                all_one_count = 0
                                homogeneous_other_count = 0
                                total_prompt_count = 0
                                generated_score_sum = 0.0
                                generated_trajectory_count = 0
                                continue
                            print(
                                f"Agent 0 uses a partial mixed prompt batch of "
                                f"{selected_prompt_count}/{prompt_batch_size}."
                            )

                        batch = self._select_prompt_groups(
                            accumulated_batch,
                            selected_prompt_count,
                        )
                    else:
                        batch = new_batch
                        mixed_prompt_count = len(group_result.mixed_uids)

                    metrics.update(
                        {
                            "rollout/all_zero_prompt_count": float(all_zero_count),
                            "rollout/all_one_prompt_count": float(all_one_count),
                            # Preserve the original ReMA names for comparable W&B plots.
                            "rollout/all_negative_cnt": float(all_zero_count),
                            "rollout/all_positive_cnt": float(all_one_count),
                            "rollout/homogeneous_other_prompt_count": float(
                                homogeneous_other_count
                            ),
                            "rollout/mixed_prompt_count": float(mixed_prompt_count),
                            "rollout/total_prompt_count": float(total_prompt_count),
                            "rollout/total_prompt_cnt": float(total_prompt_count),
                            "rollout/mixed_prompt_rate": (
                                float(mixed_prompt_count) / max(total_prompt_count, 1)
                            ),
                            "rollout/selected_prompt_count": float(selected_prompt_count),
                            "rollout/num_gen_batches": float(num_gen_batches),
                            "rollout/generated_acc": (
                                generated_score_sum / max(generated_trajectory_count, 1)
                            ),
                            "train/acc": batch.batch["acc"].float().mean().item(),
                        }
                    )

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer("adv", timing_raw):
                        if not self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    if self.config.trainer.get("save_train_generations", False):
                        self._save_train_generations(batch)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                logger.log(data=metrics, step=self.global_steps)

                accumulated_batch = None
                mixed_prompt_count = 0
                num_gen_batches = 0
                all_zero_count = 0
                all_one_count = 0
                homogeneous_other_count = 0
                total_prompt_count = 0
                generated_score_sum = 0.0
                generated_trajectory_count = 0

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return
                self.global_steps += 1
