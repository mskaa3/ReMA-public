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
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def _write_generation_checkpoint(dataset, output_lst, completed_count, output_path):
    """Atomically persist the completed dataset prefix for safe resumption."""
    completed_count = int(completed_count)
    checkpoint = dataset.iloc[:completed_count].copy()
    checkpoint['responses'] = [
        [output_lst[sample_idx][row_idx] for sample_idx in range(len(output_lst))]
        for row_idx in range(completed_count)
    ]

    output_dir = os.path.dirname(output_path)
    makedirs(output_dir, exist_ok=True)
    temporary_path = f'{output_path}.tmp.{os.getpid()}'
    checkpoint.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    print(
        f'Generation checkpoint: {completed_count}/{len(dataset)} examples '
        f'written to {output_path}'
    )


def _load_generation_checkpoint(dataset, output_path, n_samples, prompt_key):
    """Restore completed responses from an earlier generation attempt."""
    output_lst = [[] for _ in range(n_samples)]
    if not os.path.isfile(output_path):
        return output_lst, 0

    checkpoint = pd.read_parquet(output_path)
    if 'responses' not in checkpoint.columns:
        raise ValueError(f'Generation checkpoint {output_path} has no responses column')
    if len(checkpoint) > len(dataset):
        raise ValueError(
            f'Generation checkpoint has {len(checkpoint)} rows, but the input '
            f'dataset has only {len(dataset)}'
        )

    # A resumed file may come from S3, so reject an accidentally reused dataset.
    if len(checkpoint) and prompt_key in checkpoint.columns:
        for row_idx in {0, len(checkpoint) - 1}:
            if repr(checkpoint.iloc[row_idx][prompt_key]) != repr(dataset.iloc[row_idx][prompt_key]):
                raise ValueError(
                    f'Generation checkpoint prompt mismatch at row {row_idx}: '
                    f'{output_path}'
                )

    for attempts in checkpoint['responses'].tolist():
        if isinstance(attempts, np.ndarray):
            attempts = attempts.tolist()
        if not isinstance(attempts, (list, tuple)) or len(attempts) != n_samples:
            raise ValueError(
                f'Every generation checkpoint row must contain exactly '
                f'{n_samples} responses'
            )
        for sample_idx, response in enumerate(attempts):
            output_lst[sample_idx].append(response)

    print(
        f'Resuming generation from {output_path}: '
        f'{len(checkpoint)}/{len(dataset)} examples already complete'
    )
    return output_lst, len(checkpoint)


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    start_index = max(int(config.data.get('start_index', 0)), 0)
    max_examples = config.data.get('max_examples', None)
    end_index = len(dataset)
    if max_examples is not None:
        end_index = min(end_index, start_index + max(int(max_examples), 0))
    if start_index >= len(dataset) or end_index <= start_index:
        raise ValueError(
            f'Empty generation slice: start_index={start_index}, '
            f'max_examples={max_examples}, dataset_size={len(dataset)}'
        )
    dataset = dataset.iloc[start_index:end_index].reset_index(drop=True)
    print(
        f'Generation slice: [{start_index}, {end_index}) '
        f'({len(dataset)} examples)'
    )
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    n_samples = int(config.data.n_samples)
    resume_from_output = bool(config.data.get('resume_from_output', False))
    checkpoint_every_batches = max(
        int(config.data.get('checkpoint_every_batches', 0)),
        0,
    )
    samples_per_call = max(int(config.data.get('samples_per_call', 1)), 1)
    if n_samples % samples_per_call != 0:
        raise ValueError(
            f'data.n_samples={n_samples} must be divisible by '
            f'data.samples_per_call={samples_per_call}'
        )
    if int(config.rollout.n) != samples_per_call:
        raise ValueError(
            f'rollout.n={config.rollout.n} must equal '
            f'data.samples_per_call={samples_per_call}'
        )

    if resume_from_output:
        output_lst, completed_count = _load_generation_checkpoint(
            dataset,
            config.data.output_path,
            n_samples,
            config.data.prompt_key,
        )
    else:
        output_lst = [[] for _ in range(n_samples)]
        completed_count = 0

    if completed_count == total_samples:
        print(f'Generation already complete: {total_samples}/{total_samples} examples')
        return []

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    # Generation creates one rollout worker group, so reserving CPU capacity for
    # the default five colocated groups would make multi-GPU nodes unschedulable.
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        max_colocate_count=1,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()
    dispatch_dp_size = wg.world_size

    output_text_unpad = []
    for batch_start in range(completed_count, total_samples, config_batch_size):
        batch_idx = batch_start // config_batch_size
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_start:batch_start + config_batch_size]
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # Generate multiple candidates in one vLLM call. Besides being faster,
        # this avoids thousands of fragile CuMem sleep/wake transitions.
        for sample_start in range(0, n_samples, samples_per_call):
            output = wg.generate_sequences(data)
            # Outputs are interleaved by prompt, then by sample. Dummy prompts
            # were appended after real prompts, so their candidates are last.
            output = output[:real_batch_size * samples_per_call]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=False)

            # remove the padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            expected_count = real_batch_size * samples_per_call
            if len(output_text_unpad) != expected_count:
                raise RuntimeError(
                    f'Expected {expected_count} generated responses, got '
                    f'{len(output_text_unpad)}'
                )
            for row_idx in range(real_batch_size):
                row_offset = row_idx * samples_per_call
                for sample_offset in range(samples_per_call):
                    output_lst[sample_start + sample_offset].append(
                        output_text_unpad[row_offset + sample_offset]
                    )

        completed_count = batch_start + real_batch_size
        completed_batches = batch_idx + 1
        if (
            checkpoint_every_batches > 0
            and (
                completed_batches % checkpoint_every_batches == 0
                or completed_count == total_samples
            )
        ):
            _write_generation_checkpoint(
                dataset,
                output_lst,
                completed_count,
                config.data.output_path,
            )

    # Always materialize the complete file, including when periodic checkpoints
    # are disabled.
    _write_generation_checkpoint(
        dataset,
        output_lst,
        completed_count,
        config.data.output_path,
    )

    return output_text_unpad


if __name__ == '__main__':
    main()
