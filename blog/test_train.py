import torch
import torch.nn as nn
from test_model import GPT
from test_model import Config
from test_dataset import GPT2Dataset
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import logging
import os
import wandb
import math
import time


logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at epoch {epoch}")


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

max_lr = 6e-4
min_lr = 0.1 * max_lr
warm_steps = 10000  # we go with random numbers here for now
max_steps = 263636  # we go with random numbers here for now


def get_lr(step):
    if step < warm_steps:
        return max_lr * (step + 1) / warm_steps
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warm_steps) / (max_steps - warm_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config = Config(config)

    # Load the config file
    gpt2 = GPT(config)
    # Load the model
    gpt2 = gpt2.to(device)

    gpt2 = torch.compile(gpt2)

    # Send the model to GPU
    logging.info(f"compiled and Loaded model")

    if config.wandb.project_name:
        wandb.init(
            project=config.wandb.project_name,
            entity=config.wandb.entity,
            config=config.__dict__,
        )
        wandb.run.name = f"gpt_train-{wandb.run.id}"
        wandb.watch(gpt2, log="all" if config.wandb.log_gradients else "parameters")

    train_dataset = GPT2Dataset(
        config.seq_len, split="train", train_ratio=0.9, total_samples=10750
    )
    valid_dataset = GPT2Dataset(
        config.seq_len, split="valid", train_ratio=0.9, total_samples=10750
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    torch.set_float32_matmul_precision("high")

    logging.info(f"Loaded Data")

    optimizer = torch.optim.AdamW(
        gpt2.parameters(),
        lr=3e-4,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        fused=True,
    )

    loss_fn = nn.CrossEntropyLoss()
    total_batch_size = 524288
    grad_accum_steps = total_batch_size // (config.batch_size * config.seq_len)
    num_batches_to_process = (len(train_loader) // grad_accum_steps) * grad_accum_steps

    # gpt2, optimizer, start_epoch = load_checkpoint(gpt2, optimizer) #TODO uncomment this and comment next line
    start_epoch = 0
    global_step = start_epoch * (num_batches_to_process // grad_accum_steps)
    total_tokens_seen = global_step * config.batch_size * config.seq_len

    optimizer.zero_grad()

    for epoch in range(start_epoch, config.epochs):
        gpt2.train()
        epoch_start = time.time()
        epoch_tokens = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        step_loss = 0
        running_train_loss = 0
        for batch_idx, batch in enumerate(train_loop):
            if batch_idx >= num_batches_to_process:
                break
            if batch_idx % grad_accum_steps == 0:
                step_start = time.time()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            #######################
            #######################
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = gpt2(inputs)
            B, T, C = logits.shape
            micro_step_loss = loss_fn(logits.reshape(B * T, C), targets.reshape(B * T))
            micro_step_loss = micro_step_loss / grad_accum_steps
            step_loss = step_loss + micro_step_loss.item()
            micro_step_loss.backward()
            #######################
            #######################
            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_norm = nn.utils.clip_grad_norm_(gpt2.parameters(), 1.0)
                current_lr = get_lr(global_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step_end = time.time()
                running_train_loss = running_train_loss + step_loss
                train_loop.set_postfix(loss=step_loss)
                epoch_tokens = epoch_tokens + (B * T * grad_accum_steps)
                total_tokens_seen = total_tokens_seen + (B * T * grad_accum_steps)
                if config.wandb.project_name:
                    wandb.log(
                        {
                            "train/step_loss": step_loss,
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
                            "train/step_throughput": (B * T * grad_accum_steps)
                            / (step_end - step_start),
                        },
                        step=global_step,
                    )
                step_loss = 0
                global_step = global_step + 1

        avg_train_loss = running_train_loss / (
            num_batches_to_process / grad_accum_steps
        )
        throughput = epoch_tokens / (time.time() - epoch_start)
        train_loop.set_postfix(loss=avg_train_loss)
        print(
            f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}, Throughput :{throughput}"
        )
        logging.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        if config.wandb.project_name:
            wandb.log(
                {"train/average_throughput": throughput},
                step=global_step,
            )

        #########################################
        # Begin Validation
        ########################################

        gpt2.eval()
        running_val_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = gpt2(inputs)
                B, T, C = logits.shape
                loss = loss_fn(logits.reshape(B * T, C), targets.reshape(B * T))
                running_val_loss = running_val_loss + loss.item()

        avg_val_loss = running_val_loss / len(valid_loader)
        if config.wandb.project_name:
            wandb.log(
                {
                    "validation/avg_loss": avg_val_loss,
                    "validation/perplexity": math.exp(avg_val_loss),
                    "epoch": epoch,
                },
                step=global_step,
            )

        logging.info(f"Epoch {epoch+1} Average Val Loss: {avg_val_loss:.4f}")

        print(
            f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}"
        )

        save_checkpoint(gpt2, optimizer, epoch)


if __name__ == "__main__":
    main()
