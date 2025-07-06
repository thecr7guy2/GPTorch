import os
import wandb
import math
import time
import yaml
from tqdm import tqdm
import tiktoken

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from test_model import GPT
from test_model import Config
from test_dataset import GPT2Dataset


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    model_to_save = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


# def load_checkpoint(model, optimizer, path="checkpoint.pth"):
#     if os.path.isfile(path):
#         checkpoint = torch.load(path)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         start_epoch = checkpoint["epoch"] + 1
#         logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
#         return model, optimizer, start_epoch
#     else:
#         return model, optimizer, 0


def get_lr(step, total_training_steps, max_lr):
    warm_steps = int(0.1 * total_training_steps)
    min_lr = 0.1 * max_lr
    if step < warm_steps:
        return max_lr * (step + 1) / warm_steps
    if step > total_training_steps:
        return min_lr
    decay_ratio = (step - warm_steps) / (total_training_steps - warm_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def generate_sample_text(
    model,
    device,
    tokenizer,
    prompt_text="In a world where humans have unlimited power, Sai was",
    max_tokens=100,
    top_k=50,
    ):
    
    model.eval()
    input_ids = (
        torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        generated = input_ids
        for _ in range(max_tokens):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]

            logits_top_k, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = nn.functional.softmax(logits_top_k, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs[0], 1)].unsqueeze(0)

            generated = torch.cat((generated, next_token), dim=1)

        generated_text = tokenizer.decode(generated[0].tolist())

    return generated_text

def setup_ddp():
    """Initialize the distributed environment"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up the distributed process group"""
    dist.destroy_process_group()


def main():
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True


    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config = Config(config)

    gpt2 = GPT(config)
    gpt2 = gpt2.to(device)
    gpt2 = DDP(gpt2, device_ids=[local_rank], output_device=local_rank)
    gpt2 = torch.compile(gpt2)  # TODO research what fullggraph =True does
    tokenizer = tiktoken.get_encoding("gpt2")

    if config.wandb.project_name:
        if rank == 0:
            wandb.init(
                project=config.wandb.project_name,
                entity=config.wandb.entity,
                config=config.__dict__,
            )
            wandb.run.name = f"gpt_train-{wandb.run.id}"
            wandb.watch(
                gpt2.module, log="all" if config.wandb.log_gradients else "parameters"
            )
            gen_table = wandb.Table(columns=["step", "prompt", "output"])

    train_dataset = GPT2Dataset(
        config.seq_len, split="train", train_ratio=0.9
    )
    valid_dataset = GPT2Dataset(
        config.seq_len, split="valid", train_ratio=0.9
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_sampler = DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
    )

    assert (
        config.total_batch_size % (config.batch_size * config.seq_len * world_size) == 0
    )
    grad_accum_steps = config.total_batch_size // (
        config.batch_size * config.seq_len * world_size
    )
    num_batches_to_process = (len(train_loader) // grad_accum_steps) * grad_accum_steps
    steps_per_epoch = num_batches_to_process // grad_accum_steps
    total_training_steps = config.epochs * steps_per_epoch

    if rank == 0:
        print(grad_accum_steps)

    torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.AdamW(
        gpt2.parameters(),
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True,
    )

    loss_fn = nn.CrossEntropyLoss()

    # gpt2, optimizer, start_epoch = load_checkpoint(gpt2, optimizer) #TODO uncomment this and comment next line
    start_epoch = 0
    global_step = start_epoch * (num_batches_to_process // grad_accum_steps)
    total_tokens_seen = (
        global_step * config.batch_size * config.seq_len * grad_accum_steps
    )

    for epoch in range(start_epoch, config.epochs):
        gpt2.train()
        train_sampler.set_epoch(epoch)
        ##########################################
        if rank == 0:
            train_loop = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"
            )
        else:
            train_loop = train_loader
        ##########################################
        epoch_start = time.time()
        epoch_tokens = 0
        step_loss = torch.tensor(0.0, device=device)
        running_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        ##########################################
        for batch_idx, batch in enumerate(train_loop):
            if (
                batch_idx >= num_batches_to_process
            ): 
                break
            ##########################################

            if (batch_idx +1) % grad_accum_steps == 0:
                if rank == 0:
                    model_to_generate = (
                        gpt2.module if isinstance(gpt2, DDP) else gpt2
                    )
                    generated_text = generate_sample_text(
                        model_to_generate,
                        tokenizer,
                        device,
                    )
                    temp_table = wandb.Table(
                        columns=gen_table.columns, data=gen_table.data
                    )
                    temp_table.add_data(
                        global_step,
                        "In a world where humans have unlimited power, Sai was",
                        generated_text,
                    )
                    wandb.log({"generation/samples": temp_table}, step=global_step)
                    gen_table = temp_table

                gpt2.train()
            ##########################################
            if batch_idx % grad_accum_steps == 0:
                step_start = time.time()
            ##########################################
            inputs, targets = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            ##########################################
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = gpt2(inputs)
            B, T, C = logits.shape
            micro_step_loss = loss_fn(logits.float().reshape(B * T, C), targets.reshape(B * T))
            micro_step_loss = micro_step_loss / grad_accum_steps
            step_loss = step_loss + micro_step_loss.detach()
            if (batch_idx + 1) % grad_accum_steps == 0:
                gpt2.require_backward_grad_sync = True
            else:
                gpt2.require_backward_grad_sync = False
            micro_step_loss.backward()
            ##########################################
            if (batch_idx + 1) % grad_accum_steps == 0:
                dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
                grad_norm = nn.utils.clip_grad_norm_(gpt2.parameters(), 1.0)
                current_lr = get_lr(global_step, total_training_steps, config.max_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step_end = time.time()
                running_train_loss = running_train_loss + step_loss.item()
                if rank == 0:
                    train_loop.set_postfix(loss=step_loss.item())
                epoch_tokens = epoch_tokens + (B * T * grad_accum_steps * world_size)
                total_tokens_seen = total_tokens_seen + (
                    B * T * grad_accum_steps * world_size
                )
                ############################################
                if rank == 0:
                    if config.wandb.project_name:
                        wandb.log(
                            {
                                "train/step_loss": step_loss.item(),
                                "train/avg_loss": running_train_loss
                                / ((batch_idx + 1) // grad_accum_steps),
                                "train/perplexity": math.exp(
                                    running_train_loss
                                    / ((batch_idx + 1) // grad_accum_steps)
                                ),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": current_lr,
                                "train/total_tokens_seen": total_tokens_seen,
                                "train/time_per_step": (step_end - step_start) * 1000,
                                "train/step_throughput": (
                                    B * T * grad_accum_steps * world_size
                                )
                                / (step_end - step_start),
                            },
                            step=global_step,
                        )
                step_loss.zero_()
                global_step = global_step + 1

        avg_train_loss = running_train_loss / (
            num_batches_to_process / grad_accum_steps
        )
        throughput = epoch_tokens / (time.time() - epoch_start)
        if rank == 0:
            train_loop.set_postfix(loss=avg_train_loss)
            print(
                f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}, Throughput :{throughput}"
            )
            if config.wandb.project_name:
                wandb.log(
                    {"train/average_throughput": throughput},
                    step=global_step,
                )

        #########################################
        # Begin Validation
        ########################################

        gpt2.eval()
        running_val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                    device, non_blocking=True
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = gpt2(inputs)
                B, T, C = logits.shape
                loss = loss_fn(logits.float().reshape(B * T, C), targets.reshape(B * T))
                running_val_loss = running_val_loss + loss.detach()

        
       
        dist.all_reduce(running_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = running_val_loss.item() / (len(valid_loader) * world_size)
        if rank == 0:
            if config.wandb.project_name:
                wandb.log(
                    {
                        "validation/avg_loss": avg_val_loss,
                        "validation/perplexity": math.exp(avg_val_loss),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            print(
                f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}"
            )

            save_checkpoint(gpt2, optimizer, epoch)

    cleanup_ddp()


if __name__ == "__main__":
    main()
