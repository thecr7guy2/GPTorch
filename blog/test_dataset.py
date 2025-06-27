# from datasets import load_dataset
# import tiktoken
# import time

# start_time = time.time()

# tokenizer = tiktoken.get_encoding("gpt2")

# ds = load_dataset("andersonbcdefg/cc-stories-parquet")

# print(type(ds))
# print(ds.items())
# print(ds["train"]["text"][777])

# total_tokens = 0

# for text in ds["train"]["text"]:
#     tokens = tokenizer.encode(text)
#     total_tokens += len(tokens)

# end_time = time.time()

# print(f"The total number of tokens in this dataset : {total_tokens} and took {end_time-start_time} secs")


import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class GPT2Dataset(Dataset):
    """
    Something about GPT2Dataset
    """

    def __init__(self, seq_len):
        super(GPT2Dataset, self).__init__()
        tokenizer = tiktoken.get_encoding("gpt2")
        data = load_dataset("andersonbcdefg/cc-stories-parquet", split="train")
        data = data[:57500]
        # We do the exact same steps as we did in the above code.
        # We load the the tokenizer and the dataset from the data.
        data = "<|endoftext|>".join(text.strip() for text in data["text"])
        # We then concatenate the data and add a special word at the end of each
        # paragraph.
        self.tokens = tokenizer.encode(data, allowed_special={"<|endoftext|>"})
        # We tokenizer the entire data
        num_samples = len(self.tokens) // (seq_len + 1)
        # Given the sequence length we calculate the number of samples we will
        # get from the dataset.
        self.tokens = self.tokens[: num_samples * (seq_len + 1)]
        # We trim the outer edge of the dataset and retain a round figure.
        self.tokens = torch.tensor(self.tokens, dtype=torch.long).reshape(
            num_samples, seq_len + 1
        )
        # Reshape them.

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        x = self.tokens[idx, :-1]  # Input: all but last token
        y = self.tokens[idx, 1:]  # Target: all but first token
        return x, y
    

# train_dataset = GPT2Dataset(1024)
# print(len(train_dataset))
# train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

# tokenizer = tiktoken.get_encoding("gpt2")

# for x, y in train_loader:
#     x = x[0]  
#     y = y[0]
#     input_text = tokenizer.decode(x.tolist())
#     target_text = tokenizer.decode(y.tolist())
#     print(input_text)
#     print(target_text)
#     break

