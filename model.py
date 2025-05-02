import torch
import math
from torchinfo import summary


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_model * 4)
        self.linear_2 = torch.nn.Linear(d_model * 4, d_model)
        self.activation = torch.nn.GELU(approximate="tanh")
        self.linear_2.KARPATHY_VAR = 1

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class MHACausal(torch.nn.Module):
    def __init__(self, d_model, number_of_heads, context_len):
        super().__init__()

        self.d_model = d_model
        self.noh = number_of_heads
        self.context_len = context_len

        self.dk = self.d_model // self.noh

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.wo = torch.nn.Linear(d_model, d_model)

    # def calculate_self_attention(self, qprime, kprime, vprime, seq_len):
    #     dk = qprime.shape[-1]

    #     attention_scores = (qprime @ kprime.transpose(-2, -1)) / math.sqrt(dk)

    #     # (batch,noh,seq_length,dk) * #(batch,noh,dk,seq_len) => (batch,noh,seq_length,seq_len)

    #     attention_scores.masked_fill_(self.mask[:, :, :seq_len, :seq_len] == 0, -1e9)

    #     attention_scores = attention_scores.softmax(dim=-1)

    #     return attention_scores @ vprime

    def forward(self, q, k, v):

        qprime = self.wq(q)
        # (batch,seq_length,dmodel)
        kprime = self.wk(k)
        # (batch,seq_length,dmodel)
        vprime = self.wv(v)


        qprime = qprime.view(qprime.shape[0], qprime.shape[1], self.noh, self.dk)
        # (batch,seq_length,dmodel) =>(batch,seq_length,noh,dk)
        qprime = qprime.transpose(1, 2)
        # (batch,seq_length,noh,dk) => (batch,noh,seq_length,dk)

        kprime = kprime.view(kprime.shape[0], kprime.shape[1], self.noh, self.dk)
        kprime = kprime.transpose(1, 2)

        vprime = vprime.view(vprime.shape[0], vprime.shape[1], self.noh, self.dk)
        vprime = vprime.transpose(1, 2)

        x = torch.nn.functional.scaled_dot_product_attention(qprime, kprime, vprime, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.noh * self.dk)
        return self.wo(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model, noh, context_len):
        super().__init__()
        self.d_model = d_model
        self.noh = noh
        self.context_len = context_len
        self.ln1 = torch.nn.LayerNorm(self.d_model)
        self.ff_block = FeedForwardBlock(self.d_model)
        self.ln2 = torch.nn.LayerNorm(self.d_model)
        self.attention = MHACausal(self.d_model, self.noh, self.context_len)

    def forward(self, x):
        y = self.ln1(x)
        x = x + self.attention(y, y, y)
        x = x + self.ff_block(self.ln2(x))
        return x


class GPT2(torch.nn.Module):
    def __init__(self, d_model, vocab_size, noh,nlayers,config,context_len=1024):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.noh = noh
        self.context_len = context_len
        self.nlayers = nlayers
        self.config = config
        self.token_embeddings = torch.nn.Embedding(self.vocab_size, self.d_model)
        self.positional_embeddings = torch.nn.Embedding(self.context_len, self.d_model)
        self.MHA_modules = torch.nn.ModuleList(
            [DecoderBlock(self.d_model, self.noh, self.context_len) for _ in range(self.nlayers)]
        )
        self.lastln = torch.nn.LayerNorm(self.d_model, eps=1e-5)
        self.projection_layer = torch.nn.Linear(
            self.d_model, self.vocab_size, bias=False
        )
        self.projection_layer.weight = self.token_embeddings.weight
        self.apply(self.init_weights) 


    def init_weights(self,module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self,x):
        tokens = self.token_embeddings(x)
        positions = torch.arange(0, x.size(1), device=x.device)
        pos_emb = self.positional_embeddings(positions)
        x = tokens + pos_emb
        for i in self.MHA_modules:
            x = i(x)
        x = self.lastln(x)
        x = self.projection_layer(x)
        return x
    
    def get_model_summary(self, batch_size=1, seq_len=1024, device="cuda"):
        """Generate and return a summary of the model architecture"""
        sample_input = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        return summary(self, input_data=sample_input, depth=4, verbose=0)


