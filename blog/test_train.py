import torch
import torch.nn as nn
from test_model import GPT
from test_model import Config
from test_dataset import GPT2Dataset
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    def save_model(model):
        torch.save(model.state_dict(), "gpt2_model.pth")
        print("Model saved as gpt2_model.pth")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)


    config = Config(config)

    # Load the config file
    gpt2 = GPT(config)
    # Load the model
    gpt2 = gpt2.to(device)
    # Send the model to GPU

    train_dataset = GPT2Dataset(config.seq_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=4,pin_memory=True)


    optimizer = torch.optim.AdamW(
                gpt2.parameters(),
                lr=3e-4,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
        )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(0, config.epochs):
        gpt2.train()
        train_loop = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"
            )
        
        running_loss = 0 
        
        for batch_idx, batch in enumerate(train_loop):
            inputs,targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            # input.shape => (batch_size,seq_len), target.shape => (batch_size,seq_len)
            optimizer.zero_grad()
            logits = gpt2(inputs)
            B,T,C = logits.shape
            # logits.shape => (batch_size,seq_len,vocab_size)
            # Now there is a problem - The nn.CrossEntropyLoss Function accepts inputs in form (examples,classes) and outputs (examples)
            step_loss = loss_fn(logits.reshape(B*T,C),targets.reshape(B*T))
            train_loop.set_postfix(loss=step_loss.item())
            # That is why we reshape them
            
            step_loss.backward()
            optimizer.step()

            running_loss = running_loss + step_loss.item()

        epoch_loss = running_loss/len(train_loader)
        train_loop.set_postfix(loss=epoch_loss)
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")
        save_model(gpt2)


if __name__ == "__main__":
    main()




