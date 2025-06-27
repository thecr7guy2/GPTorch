import time
import torch
import torch.nn as nn
from test_model import GPT
from test_model import Config
from test_dataset import GPT2Dataset
from torch.utils.data import DataLoader
import yaml

def main():

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

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

    tic = time.time()
    for _ in range(10):
        inputs, targets = next(iter(train_loader))
        _ = gpt2(inputs.to(device))
    print("Average forward pass:", (time.time() - tic)/10)

if __name__ == "__main__":
    main()