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


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return model, optimizer, start_epoch
    else:
        return model, optimizer, 0


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
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
    # Send the model to GPU
    logging.info(f"Loaded model")

    train_dataset = GPT2Dataset(
        config.seq_len, split="train", train_ratio=0.9, total_samples=5750
    )
    valid_dataset = GPT2Dataset(
        config.seq_len, split="valid", train_ratio=0.9, total_samples=5750
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logging.info(f"Loaded Data")

    optimizer = torch.optim.AdamW(
        gpt2.parameters(),
        lr=3e-4,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    gpt2, optimizer, start_epoch = load_checkpoint(gpt2, optimizer)

    for epoch in range(start_epoch, config.epochs):
        gpt2.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

        running_train_loss = 0

        for batch_idx, batch in enumerate(train_loop):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            # input.shape => (batch_size,seq_len), target.shape => (batch_size,seq_len)
            optimizer.zero_grad()
            logits = gpt2(inputs)
            B, T, C = logits.shape
            # logits.shape => (batch_size,seq_len,vocab_size)
            # Now there is a problem - The nn.CrossEntropyLoss Function accepts inputs in form (examples,classes) and outputs (examples)
            step_loss = loss_fn(logits.reshape(B * T, C), targets.reshape(B * T))
            train_loop.set_postfix(loss=step_loss.item())
            # That is why we reshape them
            step_loss.backward()
            optimizer.step()

            running_train_loss = running_train_loss + step_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loop.set_postfix(loss=avg_train_loss)
        logging.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        # save_checkpoint(gpt2, optimizer, epoch)

        #########################################
        # Begin Validation
        ########################################

        gpt2.eval()
        running_val_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = gpt2(inputs)
                B, T, C = logits.shape
                loss = loss_fn(logits.reshape(B * T, C), targets.reshape(B * T))
                running_val_loss = running_val_loss + loss.item()

        avg_val_loss = running_val_loss / len(valid_loader)
        logging.info(f"Epoch {epoch+1} Average Val Loss: {avg_val_loss:.4f}")

        print(
            f"Epoch {epoch+1} — Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}"
        )

        save_checkpoint(gpt2, optimizer, epoch)


if __name__ == "__main__":
    main()