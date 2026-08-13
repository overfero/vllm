"""Worker-cls for profile_num_gpu_blocks.py ONLY: installs a fake
transport-backed PP group (correct pp_rank/world_size for real layer
partitioning via get_pp_indices(), but a dummy no-op transport - no real
network connection) so memory profiling reflects the REAL per-stage
layer count, not the pp_world_size=1 default (which would construct all
36 layers regardless of the checkpoint's actual content - this is
exactly the bug that made the first profiling attempt look like a
~14.5GB-regardless-of-layer-count OOM).
"""
import os

from vllm.v1.worker.gpu_worker import Worker


class _DummyTransport:
    def close(self):
        pass


class ProfileOnlyWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        import vllm.distributed.parallel_state as ps

        pp_rank = int(os.environ["PROFILE_PP_RANK"])
        pp_world_size = int(os.environ["PROFILE_PP_WORLD_SIZE"])

        group = object.__new__(ps.GroupCoordinator)
        group.transport = _DummyTransport()
        group.transport_prev = _DummyTransport() if pp_rank > 0 else None
        group.transport_next = _DummyTransport() if pp_rank < pp_world_size - 1 else None
        group.rank = pp_rank
        group.local_rank = self.local_rank
        group.rank_in_group = pp_rank
        group.world_size = pp_world_size
        group.ranks = list(range(pp_world_size))
        group.unique_name = f"profile_pp:{pp_rank}"
        group.device_communicator = None
        group.mq_broadcaster = None
        ps._PP = group
