import os
import time
import math
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from datasets import load_dataset
import tiktoken
import wandb

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import GPT2Dataset
from model import GPT2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a GPT-2 model")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise


def setup_wandb(rank, config):
    """Initialize Weights & Biases logging"""
    if rank == 0:
        wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            config=config,
        )
        wandb.run.name = f"gpt_train-{wandb.run.id}"
        config["wandb_run_id"] = wandb.run.id
        logger.info(f"Initialized wandb with run name: {wandb.run.name}")


def setup_ddp():
    """Initialize the distributed environment"""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(rank=rank, world_size=world_size, backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up the distributed process group"""
    dist.destroy_process_group()
    logger.info("DDP cleanup complete")


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1
):
    def lr_lambda(current_step):
        
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_factor = cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

        return decay_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(
    model, optimizer, scheduler, epoch, step, config, model_dir="checkpoints",
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "epochs"), exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "config": config,
        "wandb_run_id": wandb.run.id if wandb.run else None
    }

    checkpoint_path = os.path.join(model_dir, f"epochs/epoch_{epoch+1}.pt")
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu", scheduler=None):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)

    logger.info(f"Loaded checkpoint from epoch {epoch}, global step {step}")
    return epoch, step


def load_and_prepare_data(config, rank, world_size):
    """
    Load and prepare the dataset for training

    Args:
        config: Training configuration

    Returns:
        train_loader, train_sampler, steps_per_epoch
    """
    shards_dir = config["cache_dir"]
    logger.info(f"Loading shards from {shards_dir}")
    trainset = GPT2Dataset(
        data_dir=shards_dir, seq_len=config["context_len"])
    train_sampler = DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"]
    )

    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], sampler=train_sampler, drop_last=True,num_workers=4,pin_memory=True
    )
    steps_per_epoch = train_sampler.num_samples // config["batch_size"]
    logger.info(
        f"Each GPU will see {train_sampler.num_samples} samples, "
        f"{steps_per_epoch} batches per epoch"
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    return train_loader, train_sampler, steps_per_epoch,tokenizer


def generate_sample_text(
    model, tokenizer, device,
    prompt_text="In a world where humans have unlimited power, Sai was ",
    max_tokens=100, top_k=50
):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = input_ids
        for _ in range(max_tokens):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]

            logits_top_k, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(logits_top_k, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs[0], 1)].unsqueeze(0)

            generated = torch.cat((generated, next_token), dim=1)

        generated_text = tokenizer.decode(generated[0].tolist())

    return generated_text


def train_model(config, args):

    set_seed(config["seed"])
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    setup_wandb(rank, config)
    torch.set_float32_matmul_precision("high")


    model = GPT2(
        config["n_embd"],
        config["vocab_size"],
        config["n_heads"],
        config["n_layer"],
        config,
        config["context_len"],
    ).to(device)

   
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model = torch.compile(model)

    train_loader, train_sampler, steps_per_epoch,tokenizer = load_and_prepare_data(config, rank, world_size)

    optimizer_name = config.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["max_lr"],
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
            weight_decay=config.get("weight_decay", 0.1),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["max_lr"],
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
            weight_decay=config.get("weight_decay", 0.1),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


    total_training_steps = config["epochs"] * steps_per_epoch


    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_training_steps,
        min_lr_ratio=config["min_lr"] / config["max_lr"],
    )

    
    start_epoch = 0
    global_step = 0

    
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, args.resume, device, scheduler
        )
        if rank == 0:
            logger.info(f"Resumed training from epoch {start_epoch+1}, step {global_step}")

    
    log_interval = config.get("log_interval", 300)
    if rank == 0:
        wandb.watch(model, log="all", log_freq=log_interval)
        gen_table = wandb.Table(columns=["step", "prompt", "output"])

    
    num_grad_accum_steps = config["mega_batch_size"] // (config["batch_size"] * world_size)

    
    if rank == 0:
        logger.info("Starting training with configuration:")
        logger.info(f"World Size: {world_size}")
        logger.info(f"Batch size per GPU: {config['batch_size']}")
        logger.info(f"Effective total batch size: {config['batch_size'] * world_size * num_grad_accum_steps}")
        logger.info(f"Gradient accumulation steps: {num_grad_accum_steps}")
        logger.info(f"Total training steps: {total_training_steps}")
    
    
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)

        
        if rank == 0:
            train_loop = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        else:
            train_loop = train_loader
  
        
        optimizer.zero_grad()
        step_loss = 0.0
        epoch_loss = 0.0
        accumulation_counter = 0

        
        for batch_idx, batch in enumerate(train_loop):
            global_step = epoch * steps_per_epoch + batch_idx
            is_last_batch = (batch_idx + 1) == steps_per_epoch
            

            generate_interval = config.get("generate_interval", 500)
            if global_step % generate_interval == 0:
                if rank == 0:
                    model_to_generate = model.module if isinstance(model, DDP) else model
                    generated_text = generate_sample_text(
                        model_to_generate,
                        tokenizer,
                        device,
                    )

                    temp_table = wandb.Table(columns=gen_table.columns, data=gen_table.data)
                    temp_table.add_data(global_step, "In a world where humans have unlimited power, Sai was", generated_text)
                    wandb.log({"generation/samples": temp_table}, step=global_step)
                    gen_table = temp_table
                    
                model.train()

            
            if (global_step % num_grad_accum_steps == 0):
                step_time_start = time.time()
            
            
            input_ids = batch[0].to(device, non_blocking=True)
            target_ids = batch[1].to(device, non_blocking=True)

            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(input_ids)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1),
                )
                loss = loss / num_grad_accum_steps

           
            loss.backward()
            step_loss += loss.item()
            
            
            if ((global_step + 1) % num_grad_accum_steps == 0) or is_last_batch:
                
                grad_clip = config.get("grad_clip", 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

               
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                
                torch.cuda.synchronize()    
                step_time_end = time.time()
                step_time_ms = (step_time_end - step_time_start) * 1000

                
                if rank == 0:
                    epoch_loss += step_loss
                    current_lr = optimizer.param_groups[0]['lr']
                    accumulation_counter += 1
                    
                    
                    train_loop.set_postfix(
                        loss=f"{step_loss:.4f}",
                        lr=f"{current_lr:.3e}",
                        grad_norm=f"{grad_norm.item():.2f}",
                    )
                    
                    
                    wandb.log({
                        "train/loss": step_loss,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": grad_norm.item(),
                        "performance/step_time_ms": step_time_ms,
                    }, step=global_step)
                    
                step_loss = 0.0
    
        
        if rank == 0:
            avg_epoch_loss = epoch_loss / accumulation_counter
            logger.info(f"Epoch {epoch+1}/{config['epochs']} completed. Average loss: {avg_epoch_loss:.4f}")
            
            wandb.log({
                "train/epoch": epoch + 1,
                "train/epoch_loss": avg_epoch_loss
            }, step=global_step)
            
            
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                checkpoint_dir,
            )
            logger.info(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")

    
    if rank == 0:
        wandb.finish()
    
    cleanup_ddp()
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_model(config, args)