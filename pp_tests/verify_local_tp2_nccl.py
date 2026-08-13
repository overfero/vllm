"""Real NCCL sanity check across this machine's 2 local T4s - proves the
TP=2 half of the TP=2/PP=3 target topology works on real hardware
(separate from the PP-across-machines half, which uses transport_runtime
instead of NCCL and is proven elsewhere).
"""
import os
import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    x = torch.full((4, 4), float(rank + 1), device=f"cuda:{rank}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = sum(r + 1 for r in range(world_size))
    ok = torch.allclose(x, torch.full((4, 4), float(expected), device=f"cuda:{rank}"))
    print(f"[rank {rank}] all_reduce result matches expected ({expected}): {ok}", flush=True)

    y = torch.full((4,), float(rank), device=f"cuda:{rank}")
    gathered = [torch.zeros(4, device=f"cuda:{rank}") for _ in range(world_size)]
    dist.all_gather(gathered, y)
    gather_ok = all(torch.allclose(gathered[r], torch.full((4,), float(r), device=f"cuda:{rank}")) for r in range(world_size))
    print(f"[rank {rank}] all_gather correct per-rank values: {gather_ok}", flush=True)

    dist.barrier()
    if rank == 0:
        print("PASS" if (ok and gather_ok) else "FAIL")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
