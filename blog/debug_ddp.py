import os, time
import torch
import torch.distributed as dist

def main():
    # -------- env vars come from torchrun ----------
    rank        = int(os.environ["RANK"])
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    
    # -------- init DDP -----------------------------
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank if torch.cuda.is_available() else 0)

    # -------- first print --------------------------
    print(f"[Rank {rank}/{world_size}] hell yeah, we’re alive on GPU {local_rank}")

    # -------- barrier test -------------------------
    dist.barrier()        # wait for everyone
    if rank == 0:
        print(">> all ranks hit the first barrier")

    # -------- simple all-reduce --------------------
    t = torch.ones(1, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t)
    print(f"[Rank {rank}] after all-reduce tensor = {t.item()}")

    # -------- final sync + exit --------------------
    dist.barrier()
    if rank == 0:
        print(">> all done, peace ✌️")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()