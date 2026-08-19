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

"""Entrypoint for the standalone Agent 0 serial-solver trainer."""

import os

import hydra
import ray

from verl.trainer.main_ppo import get_custom_reward_fn


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )
    ray.get(TaskRunner.remote().run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from omegaconf import OmegaConf
        from pprint import pprint

        from verl.agent0_trainer.ppo.ray_trainer import Agent0RayPPOTrainer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.workers.reward_manager import Agent0RewardManager

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        if config.actor_rollout_ref.actor.strategy != "fsdp":
            raise NotImplementedError("Agent 0 currently supports FSDP only")
        if config.actor_rollout_ref.actor.strategy != config.critic.strategy:
            raise ValueError("Actor and critic strategies must match")

        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            if config.algorithm.use_kl_in_reward:
                raise ValueError("use_kl_in_reward is not supported by Agent 0")
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        if config.reward_model.enable:
            raise NotImplementedError("Agent 0 currently uses function-based rewards only")
        if config.reward_model.get("reward_manager", "agent0") != "agent0":
            raise ValueError("Agent 0 requires reward_model.reward_manager=agent0")

        compute_score = get_custom_reward_fn(config)
        reward_fn = Agent0RewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
        )
        val_reward_fn = Agent0RewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
        )

        trainer = Agent0RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=ResourcePoolManager(
                resource_pool_spec=resource_pool_spec,
                mapping=mapping,
            ),
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
