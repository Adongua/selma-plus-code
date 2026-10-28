"""NCCL-only compatibility for CrypTen's initialization PRNG seed exchange."""

import torch


def install_nccl_seed_sync_compat(crypten):
    """Move CrypTen's initialization seed tensors onto CUDA when using NCCL.

    CrypTen creates ``next_seed`` and ``global_seed`` on CPU, then sends them
    through its configured process group. NCCL accepts CUDA tensors only. The
    neighbor-only seed exchange and global broadcast are preserved; this patch
    affects initialization, not the benchmarked MPC operators.
    """
    import crypten.communicator as comm

    upstream_sync = crypten._sync_seeds

    def sync_seeds(next_seed, local_seed, global_seed):
        communicator = comm.get()
        if "nccl" not in str(communicator.get_distributed_backend()).lower():
            return upstream_sync(next_seed, local_seed, global_seed)

        device = torch.device("cuda", torch.cuda.current_device())
        next_seed = next_seed.to(device=device)
        global_seed = global_seed.to(device=device)
        prev_seed = torch.zeros(1, dtype=torch.long, device=device)

        world_size = communicator.get_world_size()
        rank = communicator.get_rank()
        if world_size >= 2:
            next_rank = (rank + 1) % world_size
            prev_rank = (next_rank - 2) % world_size
            send_request = communicator.isend(next_seed, next_rank)
            recv_request = communicator.irecv(prev_seed, src=prev_rank)
            send_request.wait()
            recv_request.wait()
        else:
            prev_seed = next_seed

        prev_value = prev_seed.item()
        next_value = next_seed.item()
        global_value = communicator.broadcast(global_seed, 0).item()

        for generator_device in crypten.generators["prev"]:
            crypten.generators["prev"][generator_device].manual_seed(prev_value)
            crypten.generators["next"][generator_device].manual_seed(next_value)
            crypten.generators["local"][generator_device].manual_seed(local_seed)
            crypten.generators["global"][generator_device].manual_seed(global_value)

    crypten._sync_seeds = sync_seeds
