import torch
from torch.utils.data import Dataset
import json
import logging

logger = logging.getLogger(__name__)

class GPT2Dataset(Dataset):
    """
    Dataset for training GPT-2 models with causal language modeling objective.
    
    This dataset takes text data, tokenizes it, and creates input-target pairs
    where each target is the input shifted by one token to the right.
    
    Args:
        data (str): The raw text data to be tokenized
        seq_len (int): Maximum sequence length for model inputs
        tokenizer: A tokenizer object that has an encode method
    """
    def __init__(self, data, seq_len, tokenizer):
        super().__init__()
        
        if not data:
            raise ValueError("Input data cannot be empty")
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")
        if not hasattr(tokenizer, 'encode'):
            raise ValueError("Tokenizer must have an 'encode' method")
            
        self.seq_len = seq_len
        self.data = data[:5000000]
        self.tokenizer = tokenizer

        logger.info(f"Tokenizing dataset with sequence length {seq_len}")
        self.tokens = self.tokenizer.encode(self.data, allowed_special={'<|endoftext|>'})
        logger.info(f"Total tokens: {len(self.tokens)}")

        # Calculate number of samples and reshape data
        num_samples = len(self.tokens) // (self.seq_len + 1) 
        self.tokens = self.tokens[: num_samples * (self.seq_len + 1)]
        self.tokens = torch.tensor(self.tokens, dtype=torch.long).reshape(num_samples, self.seq_len + 1)
        logger.info(f"Created {num_samples} training samples")


    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx, :-1]  # Input: all but last token
        y = self.tokens[idx, 1:]   # Target: all but first token
        return x, y