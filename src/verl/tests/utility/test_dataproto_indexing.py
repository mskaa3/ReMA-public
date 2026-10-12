import numpy as np
import pytest
import torch

from verl.protocol import DataProto, DataProtoItem


def make_batch(size=6):
    return DataProto.from_dict(
        tensors={"row_id": torch.arange(size), "labels": torch.arange(size * 2).reshape(size, 2)},
        non_tensors={
            "uid": [f"q{i // 32}" for i in range(size)],
            "history": [{"row_id": i} for i in range(size)],
        },
        meta_info={"c3_focal_role": "decomposer"},
    )


@pytest.mark.parametrize("index,expected", [
    (slice(1, 5, 2), [1, 3]),
    ([4, 1, 4], [4, 1, 4]),
    ([2], [2]),
    ([], []),
    (np.array([4, 1]), [4, 1]),
    (np.array([], dtype=np.int64), []),
    (torch.tensor([4, 1]), [4, 1]),
    (torch.tensor([2]), [2]),
    (torch.tensor([], dtype=torch.long), []),
    (np.array([False, True, False, True, False, False]), [1, 3]),
    (torch.tensor([False, True, False, True, False, False]), [1, 3]),
    (torch.zeros(6, dtype=torch.bool), []),
])
def test_batch_indices_preserve_dataproto_and_row_alignment(index, expected):
    source = make_batch()
    selected = source[index]

    assert isinstance(selected, DataProto)
    assert len(selected) == len(expected)
    assert selected.batch.batch_size == torch.Size([len(expected)])
    assert selected.batch["row_id"].tolist() == expected
    assert [row["row_id"] for row in selected.non_tensor_batch["history"]] == expected
    assert selected.non_tensor_batch["history"].dtype == object
    assert selected.meta_info is source.meta_info
    assert isinstance(selected[:1], DataProto)
    assert len(source) == 6


@pytest.mark.parametrize("index", [2, np.int64(2), np.array(2), torch.tensor(2)])
def test_scalar_indices_keep_dataprotoitem_for_reward_and_dataloader(index):
    selected = make_batch()[index]

    assert isinstance(selected, DataProtoItem)
    assert selected.batch.batch_size == torch.Size([])
    assert selected.batch["row_id"].item() == 2
    assert selected.non_tensor_batch["history"] == {"row_id": 2}


@pytest.mark.parametrize("accumulate", [False, True])
def test_group_filter_can_select_then_trim_complete_c3_groups(accumulate):
    source = make_batch(96)
    kept_uids = {"q0", "q2"}
    kept_traj_idxs = [
        i for i, uid in enumerate(source.non_tensor_batch["uid"]) if uid in kept_uids
    ]
    batch = source[kept_traj_idxs]
    if accumulate:
        # Sparse generation batches may contribute no eligible groups.
        batch = DataProto.concat([source[[]], batch])

    batch = batch[:32]

    assert isinstance(batch, DataProto)
    assert batch.batch["row_id"].tolist() == list(range(32))
    assert batch.non_tensor_batch["uid"].tolist() == ["q0"] * 32
    assert len(list(batch.make_iterator(mini_batch_size=16, epochs=1))) == 2


def test_metadata_only_batch_can_be_indexed():
    source = DataProto(non_tensor_batch={"uid": np.array(["a", "b", "c"], dtype=object)})

    assert isinstance(source[1], DataProtoItem)
    assert source[1].non_tensor_batch["uid"] == "b"
    selected = source[[2, 0]]
    assert isinstance(selected, DataProto)
    assert selected.batch is None
    assert selected.non_tensor_batch["uid"].tolist() == ["c", "a"]
    assert len(source[[]]) == 0
