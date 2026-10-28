"""NCCL-only compatibility for CrypTen's initialization PRNG seed exchange."""

import json
import torch
import torch.distributed as dist


def install_nccl_seed_sync_compat(crypten):
    """Use an initialization-only Gloo group for CrypTen's CPU seed tensors.

    CrypTen creates ``next_seed`` and ``global_seed`` on CPU, then sends them
    through its configured process group. NCCL accepts CUDA tensors only, and
    moving these tiny startup messages to NCCL P2P can hang during communicator
    setup. This preserves CrypTen's original CPU messages and neighbor-only
    exchange using a temporary Gloo subgroup. MPC operators continue to use
    the configured NCCL group.
    """
    import crypten.communicator as comm

    upstream_sync = crypten._sync_seeds

    def sync_seeds(next_seed, local_seed, global_seed):
        communicator = comm.get()
        if "nccl" not in str(communicator.get_distributed_backend()).lower():
            return upstream_sync(next_seed, local_seed, global_seed)

        world_size = communicator.get_world_size()
        rank = communicator.get_rank()
        if rank == 0:
            print(json.dumps({"event": "seed_gloo_group_create_started"}), flush=True)
        seed_group = dist.new_group(
            ranks=list(range(world_size)), backend="gloo"
        )
        if rank == 0:
            print(json.dumps({"event": "seed_gloo_group_create_complete"}), flush=True)
        prev_seed = torch.zeros(1, dtype=torch.long)
        if world_size >= 2:
            next_rank = (rank + 1) % world_size
            prev_rank = (next_rank - 2) % world_size
            send_request = dist.isend(next_seed, dst=next_rank, group=seed_group)
            recv_request = dist.irecv(prev_seed, src=prev_rank, group=seed_group)
            send_request.wait()
            recv_request.wait()
        else:
            prev_seed = next_seed
        if rank == 0:
            print(json.dumps({"event": "seed_neighbor_exchange_complete"}), flush=True)

        prev_value = prev_seed.item()
        next_value = next_seed.item()
        dist.broadcast(global_seed, src=0, group=seed_group)
        global_value = global_seed.item()
        if rank == 0:
            print(json.dumps({"event": "seed_broadcast_complete"}), flush=True)
        dist.destroy_process_group(seed_group)

        for generator_device in crypten.generators["prev"]:
            crypten.generators["prev"][generator_device].manual_seed(prev_value)
            crypten.generators["next"][generator_device].manual_seed(next_value)
            crypten.generators["local"][generator_device].manual_seed(local_seed)
            crypten.generators["global"][generator_device].manual_seed(global_value)

    crypten._sync_seeds = sync_seeds
