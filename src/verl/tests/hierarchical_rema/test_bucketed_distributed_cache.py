"""Real three-rank CPU test of the length-bucketed old-policy cache."""
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from test_training_throughput import offline, DistributedProbeModel, probe_rows


class ReplayRows(list):
    @property
    def rows(self):
        return self


def bucketed_probe(rank, directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(Path(directory) / "rendezvous").as_uri(),
                            rank=rank, world_size=3, timeout=timedelta(seconds=45))
    try:
        model = torch.nn.parallel.DistributedDataParallel(DistributedProbeModel())
        context = offline.DistributedTrainingContext(enabled=True, rank=rank, world_size=3, backend="gloo")
        values = offline._compute_old_log_prob_cache(model, ReplayRows(probe_rows()), 2, "cpu", context,
                                                     length_bucketing=True)
        computed_rows = model.module.count
        # One rank has no scored tokens, but must still join DDP forwards.
        validation_rows = probe_rows()[:2]
        if rank == 0:
            for row in validation_rows:
                row["loss_mask"].zero_()
        loader = torch.utils.data.DataLoader(validation_rows, batch_size=2, collate_fn=offline._collate_rows)
        metrics = offline.evaluate_controller_model(model, loader, "cpu", context)
        torch.save({"values": values, "computed_rows": computed_rows, "metrics": metrics},
                   Path(directory) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires CPU Gloo")
def test_real_three_rank_bucketed_cache(tmp_path):
    import torch.multiprocessing as mp
    baseline = offline._compute_old_log_prob_cache(DistributedProbeModel(), probe_rows(), 2, "cpu")
    loader = torch.utils.data.DataLoader(probe_rows()[:2], batch_size=2, collate_fn=offline._collate_rows)
    expected_loss = offline.evaluate_controller_model(DistributedProbeModel(), loader, "cpu")["val_loss"]
    mp.spawn(bucketed_probe, args=(str(tmp_path),), nprocs=3, join=True)
    total = 0
    for rank in range(3):
        saved = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        total += saved["computed_rows"]
        assert saved["metrics"]["val_loss"] == pytest.approx(expected_loss)
        assert set(saved["values"]) == set(baseline)
        for index in baseline:
            torch.testing.assert_close(saved["values"][index], baseline[index], rtol=0, atol=0)
    assert total == 9
