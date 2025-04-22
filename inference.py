
import torch
from model import GPT2
import tiktoken
import yaml


def load_checkpoint(model, checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    if any(k.startswith("_orig_mod.module.") for k in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod.module.") :
                new_key = key[len("_orig_mod.module."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model

def load_checkpoint2(model, checkpoint_path,device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod.") :
                new_key = key[len("_orig_mod."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

def perform_inference(model,device,prompt_text):
    
    model.eval()

    input_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long).unsqueeze(0).to(device)

    generated_topk = input_ids.clone()

    top_k_text = ""

    with torch.no_grad(): 
        temp_generated_topk = generated_topk
        for _ in range(32):
            outputs = model(temp_generated_topk)
            next_token_logits = outputs[:, -1, :]
            kth_vals, _ = torch.topk(next_token_logits, 30, dim=-1)
            kth_val_per_batch = kth_vals[:, -1].unsqueeze(-1)
            indices_to_remove = next_token_logits < kth_val_per_batch
            filtered_logits = next_token_logits.masked_fill(indices_to_remove, -float('inf'))
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            temp_generated_topk = torch.cat((temp_generated_topk, next_token), dim=1)


        top_k_text = tokenizer.decode(temp_generated_topk[0].tolist())

        return top_k_text
    

config = load_config("config.yaml")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2(config["n_embd"], config["vocab_size"], config["n_heads"],config["n_layer"],config,config["context_len"]).to(device)
model = load_checkpoint(model,"checkpoints/epoch_1.pt",device)
prompt_text = '''I tried so hard, and got so far'''

top_k = perform_inference(model,device,prompt_text)
print(f"New models output is: {top_k}")

print("################################################################################")
model = load_checkpoint2(model,"checkpoints/cc_stories_epoch_9.pt",device)
top_k = perform_inference(model,device,prompt_text)
print(f"Old models output is: {top_k}")