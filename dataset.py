import torch
from torch.utils.data import Dataset
import logging
import os
import glob
import bisect
import numpy as np

logger = logging.getLogger(__name__)

class GPT2Dataset(Dataset):
    def __init__(self, data_dir, seq_len):
        super().__init__()
        self.seq_len = seq_len
        pattern = os.path.join(data_dir, "**", "shard_*.npy")
        self.shard_files = sorted(glob.glob(pattern, recursive=True))

        if not self.shard_files:
            raise FileNotFoundError(f"No .npy shards found under {data_dir}. "
                                   "Did you run preprocess.py successfully?")

        self.shards_mmap = []  
        self.shard_lens = []  
        self.cumsum = []       
        total_len = 0
        expected_token_len = self.seq_len + 1

        logger.info(f"Loading metadata for {len(self.shard_files)} shards...")
        for i, path in enumerate(self.shard_files):
            try:
                shard_mmap = np.load(path, mmap_mode='r')

                if shard_mmap.ndim != 2 or shard_mmap.shape[1] != expected_token_len:
                    logger.warning(f"Shard {path} has unexpected shape {shard_mmap.shape}. "
                                   f"Expected (_, {expected_token_len}). Skipping shard.")
                    continue 

                shard_len = shard_mmap.shape[0] 
                if shard_len == 0:
                    logger.warning(f"Shard {path} is empty. Skipping.")
                    continue

                self.shards_mmap.append(shard_mmap)
                self.shard_lens.append(shard_len)
                total_len += shard_len
                self.cumsum.append(total_len)

            except Exception as e:
                logger.error(f"Error loading or validating shard {path}: {e}. Skipping.")

        self.total_len = total_len
        if self.total_len == 0:
            raise ValueError("No valid data loaded after processing all shards. Check logs and shard files.")


    def __len__(self):
        """Returns the total number of sequences across all shards."""
        return self.total_len

    def __getitem__(self, idx):
        """
        Retrieves the idx-th sequence from the dataset.

        Uses bisect to find the correct shard and index within that shard,
        then accesses the data directly from the memory-mapped NumPy array.
        """
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range for dataset size {self.total_len}")

    
        shard_idx = bisect.bisect_right(self.cumsum, idx)

        # Calculate the index within the selected shard
        if shard_idx == 0:
            local_idx = idx
        else:
            # Subtract the cumulative length of previous shards
            local_idx = idx - self.cumsum[shard_idx - 1]

        try:
            token_seq_np = self.shards_mmap[shard_idx][local_idx]
        except IndexError:
             # This should ideally not happen if cumsum and shard_lens are correct
             logger.error(f"Internal Error: Calculated local_idx {local_idx} out of bounds "
                          f"for shard {shard_idx} (len {self.shard_lens[shard_idx]}) with global idx {idx}.")
             raise IndexError("Internal dataset indexing error.")


        x = torch.from_numpy(token_seq_np[:-1].astype(np.int64))
        y = torch.from_numpy(token_seq_np[1:].astype(np.int64))

        return x, y