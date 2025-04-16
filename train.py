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

from dataset import GPT2Dataset
from model import GPT2


# Configure logging
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
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
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


def setup_wandb(config, resume_id=None):
    """Initialize Weights & Biases for experiment tracking"""
    try:
        if resume_id:
            wandb.init(
                project=config["wandb_project"],
                config=config,
                id=resume_id,
                resume="must",
            )
        else:
            run_name = config.get("run_name", None)
            if run_name:
                run_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            else:
                run_name = f"gpt2-{config['n_layer']}-{config['n_embd']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            wandb.init(project=config["wandb_project"], config=config, name=run_name)
        logger.info(f"Initialized W&B run: {wandb.run.name}")
    except Exception as e:
        logger.error(f"Error setting up W&B: {e}")
        logger.warning("Continuing without W&B logging")


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
    """
    Create a learning rate scheduler with warmup and cosine decay

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a fraction of max_lr
        last_epoch: The index of the last epoch

    Returns:
        PyTorch LR scheduler
    """

    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decay_factor = cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

        return decay_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    config,
    model_dir="checkpoints",
):
    """
    Save model checkpoint

    Args:
        model: The model to save
        optimizer: The optimizer to save
        scheduler: The scheduler to save
        epoch: Current epoch number
        step: Current global step
        loss: Current loss value
        config: Training configuration
        model_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far

    Returns:
        Path to the saved checkpoint
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "epochs"), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": config, 
    }

    # Save regular epoch checkpoint
    checkpoint_path = os.path.join(model_dir, f"epochs/epoch_{epoch+1}.pt")
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu", scheduler=None):
    """
    Load a model checkpoint

    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint to
        scheduler: The scheduler to load state into

    Returns:
        Tuple of (epoch, step, run_id) from the checkpoint
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats (compiled vs non-compiled models)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        # Handle compiled model checkpoint
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod.") :]
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

    # Try to get W&B run ID for resuming
    run_id = None
    if "wandb_run_id" in checkpoint:
        run_id = checkpoint["wandb_run_id"]

    logger.info(f"Loaded checkpoint from epoch {epoch}, global step {step}")
    return epoch, step, run_id


def load_and_prepare_data(config):
    """
    Load and prepare the dataset for training

    Args:
        config: Training configuration

    Returns:
        DataLoader and tokenizer
    """
    # Load tokenizer
    tokenizer_name = config.get("tokenizer", "gpt2")
    try:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        logger.info(f"Loaded tokenizer: {tokenizer_name}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

    # Load dataset
    cache_dir = config.get("cache_dir", None)
    dataset_name = config.get("dataset_name")

    try:
        logger.info(f"Loading dataset: {dataset_name}")
        data = load_dataset(dataset_name, cache_dir=cache_dir, split="train")

        # Join all texts with EOS token
        logger.info("Preparing dataset...")
        concat_data = "<|endoftext|>".join(text.strip() for text in data["text"])

        # Create dataset
        dataset = GPT2Dataset(concat_data, config["context_len"], tokenizer)

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        logger.info(f"Created dataloader with {len(loader)} batches")
        return loader, tokenizer

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        raise


def generate_sample_text(
    model, tokenizer, device, prompt_text="In a world where humans have unlimited power, Sai was ", max_tokens=50,):
    """
    Generate sample text from the model

    Args:
        model: The model to generate from
        tokenizer: The tokenizer to use
        device: Device to run generation on
        prompt_text: Text prompt to start generation
        max_tokens: Maximum number of tokens to generate
        config: Optional generation config

    Returns:
        Generated text
    """
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = input_ids
        for _ in range(max_tokens):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]  # get logits for the last token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat((generated, next_token), dim=1)
        
        generated_text = tokenizer.decode(generated[0].tolist())

    return generated_text  

   


def train_model(config, args):
    """
    Main training function

    Args:
        config: Training configuration
        args: Command line arguments
    """
    # Set up training environment
    set_seed(config["seed"])

    if not args.no_wandb and "wandb_project" in config:
        setup_wandb(config)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision("high")


    logger.info(f"Using device: {device}")

    model = GPT2(
        config["n_embd"],
        config["vocab_size"],
        config["n_heads"],
        config["n_layer"],
        config,
        config["context_len"],
    ).to(device)

    model = torch.compile(model)

        

    # Log model summary
    if config.get("log_model_summary", False):
        logger.info(f"Model summary:\n{model.get_model_summary(device=device)}")

    # Load data
    train_loader, tokenizer = load_and_prepare_data(config)

    # Setup optimizer

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["max_lr"],
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
            weight_decay=config.get("weight_decay", 0.1),
    )

    # Calculate training steps
    steps_per_epoch = len(train_loader)
    total_training_steps = config["epochs"] * steps_per_epoch

    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_training_steps,
        min_lr_ratio=config["min_lr"] / config["max_lr"],
    )

    gen_table = wandb.Table(columns=["step", "prompt", "output"])

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if args.resume:
        start_epoch, global_step, run_id = load_checkpoint(
            model, optimizer, args.resume, device, scheduler
        )
        # Resume W&B if specified
        if not args.no_wandb and run_id:
            setup_wandb(config, resume_id=run_id)

    # Track W&B run ID for resumption
    if not args.no_wandb:
        config["wandb_run_id"] = wandb.run.id

    # Setup W&B model watching and generation table
    if not args.no_wandb:
        log_interval = config.get("log_interval", 300)
        wandb.watch(model, log="all", log_freq=log_interval)
        gen_table = wandb.Table(columns=["step", "prompt", "output"])

    # Calculate gradient accumulation steps
    num_grad_accum_steps = config["mega_batch_size"] // config["batch_size"]

    # Create checkpoint directory
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"
        )

        
        accumulated_loss = 0.0
        optimizer.zero_grad()
        accumulation_counter = 0

        for batch_idx, batch in enumerate(train_loop):
            global_step = epoch * steps_per_epoch + batch_idx
            generate_interval = config.get("generate_interval", 500)
            if global_step % generate_interval == 0:
                generated_text = generate_sample_text(
                        model,
                        tokenizer,
                        device,
                    )

                if not args.no_wandb:
                    temp_table = wandb.Table(columns=gen_table.columns, data=gen_table.data)
                    temp_table.add_data(global_step, "In a world where humans have unlimited power, Sai was", generated_text)
                    wandb.log({"generation/samples": temp_table}, step=global_step)
                    gen_table = temp_table
                    
                model.train()

            input_ids = batch[0].to(device)
            target_ids = batch[1].to(device)


            with (torch.autocast(device_type=device.type, dtype=torch.bfloat16)):
                outputs = model(input_ids)
                    
                loss = F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        target_ids.view(-1),
                    )
                   
                loss = loss / num_grad_accum_steps

        
            loss.backward()

            accumulated_loss = accumulated_loss + loss.item()

            accumulation_counter += 1
            

            if accumulation_counter >= num_grad_accum_steps:
                grad_clip = config.get("grad_clip", 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                log_interval =  num_grad_accum_steps
                if global_step % log_interval == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    average_loss = accumulated_loss

                    train_loop.set_postfix(
                        loss=f"{average_loss:.4f}",
                        lr=f"{current_lr:.3e}",
                        grad_norm=f"{grad_norm.item():.2f}",
                    )

                    if not args.no_wandb:
                        wandb.log(
                            {
                                "train/loss": average_loss,
                                "train/perplexity": math.exp(average_loss),
                                "train/learning_rate": current_lr,
                                "train/grad_norm": grad_norm.item(),
                            },
                            step=global_step,
                        )
                accumulated_loss = 0.0 
                accumulation_counter = 0 


        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            config,
            checkpoint_dir,
        )

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    # End of training
    logger.info(f"Training completed after {config['epochs']} epochs")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_model(config, args)
