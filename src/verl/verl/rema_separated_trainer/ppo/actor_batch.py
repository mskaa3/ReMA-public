"""Loss-neutral transport padding, applied only after C3 estimation."""

import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor


def pad_scoped_actor_batch(batch: DataProto, world_size: int):
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    trainable = batch.batch['labels'].ne(-100).any(dim=-1).sum().item()
    # Padding cannot manufacture a real trainable action for an empty rank.
    if not len(batch) or trainable < world_size:
        return batch, 0
    padded, padding = pad_dataproto_to_divisor(batch, world_size)
    if not padding:
        return batch, 0
    mask = torch.zeros(len(padded), dtype=torch.bool, device=padded.batch['labels'].device)
    mask[-padding:] = True
    padded.batch['actor_padding_mask'] = mask
    padded.batch['labels'][-padding:] = -100
    padded.batch['step_ids'][-padding:] = -100
    for key in ('advantages', 'returns', 'token_level_scores', 'token_level_rewards'):
        if key in padded.batch:
            padded.batch[key][-padding:] = 0
    return padded, padding


def without_actor_padding(batch: DataProto) -> DataProto:
    if 'actor_padding_mask' not in batch.batch:
        return batch
    keep = ~batch.batch['actor_padding_mask']
    return DataProto(
        batch=batch.batch[keep],
        non_tensor_batch={k: v[keep.cpu().numpy()] for k, v in batch.non_tensor_batch.items()},
        meta_info=batch.meta_info,
    )


def actor_loss_denominators(labels, step_ids, agg_mode):
    """Policy and regularizer denominators, excluding masked actions."""
    mask = labels.ne(-100)
    tokens = mask.sum()
    if agg_mode == 'token':
        policy = tokens
    elif agg_mode == 'trajectory':
        policy = mask.any(-1).sum()
    elif agg_mode == 'turn':
        rows = torch.arange(len(labels), device=labels.device)[:, None].expand_as(labels)
        turns = torch.stack((rows[mask], step_ids[mask]), dim=-1).unique(dim=0)
        policy = tokens.new_tensor(len(turns))
    else:
        raise ValueError(f'Unknown loss aggregation: {agg_mode}')
    return torch.stack((policy, tokens))


def padded_microbatch_loss_scale(microbatch_count, global_minibatch_count, world_size):
    """Compensate DDP averaging using actual loss denominators, not padding."""
    return world_size * microbatch_count / global_minibatch_count.clamp(min=1)
