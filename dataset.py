import torch
from torch.utils.data import Dataset
import logging
import os
import glob
import bisect



class GPT2Dataset(Dataset):
    def __init__(self, data_dir, seq_len):
        super().__init__()
        self.seq_len = seq_len
        pattern = os.path.join(data_dir, "**", "shard_*.pt")
        self.shard_files = sorted(glob.glob(pattern, recursive=True))
        if not self.shard_files:
            raise ValueError(f"No .pt shards found under {data_dir}")
        
        self.shard_lens = []
        self.cumsum = []
        total = 0
        for path in self.shard_files:
            shard_data = torch.load(path, map_location="cpu")
            length = len(shard_data)
            self.shard_lens.append(length)
            total += length
            self.cumsum.append(total)
            del shard_data
        self.total_len = total

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        
        shard_idx = bisect.bisect_right(self.cumsum, idx)
        num_before = self.cumsum[shard_idx - 1] if shard_idx > 0 else 0
        local_idx = idx - num_before

        
        shard = torch.load(self.shard_files[shard_idx], map_location="cpu")
        token_seq = shard[local_idx]
        
        x = token_seq[:-1]
        y = token_seq[1:]
        return x, y