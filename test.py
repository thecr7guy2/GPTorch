import os
import tiktoken
import torch

def inspect_shard(shard_path, context_length=1024, num_samples=3):
    """Inspect a shard file and validate its contents"""

    tokenizer = tiktoken.get_encoding("gpt2")
    
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    
    data = torch.load(shard_path)

    print(f"Loaded shard with {len(data)} chunks")
    print(f"chunk shape is {data[0].shape}")
    # Validation checks
    total_tokens = 0
    valid_chunks = 0
    
    for i, chunk in enumerate(data):
        if chunk.shape != (context_length + 1,):
            print(f"Invalid chunk shape at index {i}: {chunk.shape}")
            continue
            
        # Check token values
        tokens = chunk.tolist()
        if any(not isinstance(t, int) or t < 0 for t in tokens):
            print(f"Invalid token values in chunk {i}")
            continue
            
        total_tokens += len(tokens)
        valid_chunks += 1
        
        # Print sample decodes
        if i < num_samples:
            text = tokenizer.decode(tokens[:context_length])
            next_token = tokenizer.decode([tokens[-1]])
            print(f"\nSample {i+1}:")
            print(f"Context text: {text}")
            print(f"Target token: {next_token}")
            print("-" * 50)
    
    # Shard statistics
    print(f"\nShard Validation Summary:")
    print(f"Total chunks: {len(data)}")
    print(f"Valid chunks: {valid_chunks}")
    print(f"Total tokens: {total_tokens}")
    print(f"Expected chunk size: {context_length + 1}")
    print(f"Actual tokens in shard: {total_tokens}")


inspect_shard("./data/Stack/shard_0_pid4104.pt")