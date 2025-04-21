import os
import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm
import yaml


TOKENIZER = tiktoken.get_encoding("gpt2")
os.environ["HF_TOKEN"] = "ADD TOKEN HERE"

def process_shard(dataset_config, cache_dir):
    """Process a dataset subset with configurable limits"""
    context_len = dataset_config.get("context_len", 1024)
    chunk_len = context_len + 1
    max_chunks_per_shard = (100_000_000 // chunk_len ) -1  
    max_tokens = dataset_config.get("max_tokens", None)
    total_processed = 0
    shard_idx = 0
    buffer = []
    
    output_dir = os.path.join(cache_dir, dataset_config['name'])
    os.makedirs(output_dir, exist_ok=True)

    kwargs = dataset_config.get("kwargs", {})
    
    ds = load_dataset(
        dataset_config["hf_path"],
        split=dataset_config.get("split", "train"),
        streaming=True,
        trust_remote_code=True,
        **kwargs
    )
    
    for example in tqdm(ds, desc=dataset_config["name"]):
        text = example.get(dataset_config.get("text_field", "text"), "")
        tokens = TOKENIZER.encode(
            text + "<|endoftext|>",
            allowed_special={'<|endoftext|>'}
        )
        
        for i in range(0, len(tokens), chunk_len):
            chunk = tokens[i:i + chunk_len]
            if len(chunk) == chunk_len:
                buffer.append(torch.tensor(chunk, dtype=torch.long))
                total_processed += chunk_len
                
                
                if len(buffer) == max_chunks_per_shard:
                    torch.save(buffer, os.path.join(output_dir, f"shard_{shard_idx}.pt"))
                    buffer = []
                    shard_idx += 1
                
               
                if max_tokens and total_processed >= max_tokens:
                    break 
        
        
        if max_tokens and total_processed >= max_tokens:
            break  
    

    if buffer:
        torch.save(buffer, os.path.join(output_dir, f"shard_{shard_idx}.pt"))
    
    return total_processed

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    total_tokens = 0
    target_tokens = config["dataset"].get("target_tokens", 5_000_000_000)

    for subset in config["dataset"]["subsets"]:
        processed = process_shard(subset, config["dataset"]["cache_dir"])
        total_tokens += processed
        print(f"Processed {processed:,} tokens for {subset['name']}")
        
        if total_tokens >= target_tokens:
            print(f"Reached global target of {target_tokens:,} tokens")
            break

    print(f"Total processed: {total_tokens:,} tokens")