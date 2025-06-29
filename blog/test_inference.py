import tiktoken
import torch
import torch.nn as nn
import yaml
from test_model import GPT



class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
config = Config(config)
sample_input = torch.randint(0, 100, (4,1024)).to(device)
gpt2 = GPT(config)
gpt2.load_state_dict(torch.load("checkpoint.pth", map_location=device)["model_state_dict"])
gpt2 = gpt2.to(device)

gpt2.eval()

tokenizer = tiktoken.get_encoding("gpt2")

prompt = "I am sai and I like"
input_ids = torch.tensor(tokenizer.encode(prompt)).to(device).unsqueeze(0)

# for _ in range(20): 
#     with torch.no_grad():
#         outputs = gpt2(input_ids)
#     next_token_logits = outputs[:, -1, :]
#     next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
#     input_ids = torch.cat((input_ids, next_token), dim=1)

for _ in range(20): 
    with torch.no_grad():
        outputs = gpt2(input_ids)
        #outputs.shape(1,seq_len,50304)
    next_token_logits = outputs[:, -1, :]
    # We only want the last tokens predictions
    # So we set the seq_len dimension to -1.
    # We now select the top-50 most probable tokens
    values, indices = torch.topk(next_token_logits, 50)
    probs = nn.functional.softmax(values, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    next_token = indices.gather(-1, next_token)
    input_ids = torch.cat((input_ids, next_token), dim=1)

print(tokenizer.decode(input_ids.cpu().flatten().tolist()))
# Decode it.

