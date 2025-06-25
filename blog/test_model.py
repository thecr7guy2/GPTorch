import torch
import torch.nn as nn
import math
import yaml
from torchinfo import summary


class MLP(nn.Module):
    """Some Information about MLP"""
    def __init__(self,config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd,config.n_embd*4)
        self.c_proj = nn.Linear(config.n_embd*4,config.n_embd)
        self.activation = nn.GELU(approximate='tanh')

    def forward(self, x):
        x = self.c_fc(x)
        # here x is (4,1024,768)
        # (4,1024,3072) = (4,1024,768) * (768,3072) 
        x = self.activation(x)
        x = self.c_proj(x)
        # (4,1024,768) = (4,1024,3072) * (3072 * 768)
        return x


class MultiHeadedAttention(nn.Module):
    """Some Information about MultiHeadedAttention"""
    def __init__(self,config):
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd,config.n_embd*3)
        # self.c_attn.weights.shape -> (768,2304)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        # self.c_proj.weights.shape -> (768,768)
        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                               .view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x):
		# Here the input (x) is (B,T,C) <-> (4,1024,768)
        qkv = self.c_attn(x)
        # qkv.shape -> (4, 1024, 2304)
        # (4,1024,2034) = (4,1024,768) * (768*2304)
        # We now want to split this matrix into 3 matrices 
        q,k,v = qkv.split(self.config.n_embd,dim=2)
        # q.shape -> (4,1024,768) 
        # k.shape -> (4,1024,768)
        # v.shape -> (4,1024,768)
        #################################################
        # We need to take care about multi headed attention
        ##################################################
        qprime = q.reshape(x.shape[0],x.shape[1],self.config.n_heads,self.config.n_embd//self.config.n_heads).transpose(1,2)
        # you could also use `view` here. The only thing is view requires a contiguous tensor
        kprime = k.reshape(x.shape[0],x.shape[1],self.config.n_heads,self.config.n_embd//self.config.n_heads).transpose(1,2)
        vprime = v.reshape(x.shape[0],x.shape[1],self.config.n_heads,self.config.n_embd//self.config.n_heads).transpose(1,2)
        ####################################################
        # qprime.shape -> (4,12,1024,64)
        # vprime.shape -> (4,12,1024,64)
        # kprime.shape -> (4,12,1024,64)
        ####################################################
        # Now we calculate the attention scores.
        attention_scores = (qprime @ kprime.transpose(2,3))/(math.sqrt(qprime.shape[-1]))
        # (4,12,1024,64) * (4,12,64,1024) => (4,12,1024,1024)
        # Now we have attention scores of each 12 heads - Attention Scores is nothing but a square matrix.
        # For each 1024 tokens in the sequence we create a square matrix and this gives us how much attention 
        # each token has to pay to each other.
        ##############################################
        # We can now use this attention scores to visualize if the transformer is working correctly or not.
        ##############################################
        # Now we make this attention causal - Means we make sure that the tokens cannot see the future tokens.
        attention_scores_masked = attention_scores.masked_fill(self.bias[:,:,:x.shape[1],:x.shape[1]]==0,float("-inf"))
        attention_scores = nn.functional.softmax(attention_scores_masked,dim=-1)
        # attention_scores -> (B,nh,T,T) -> (4,12,1024,1024)
        y = attention_scores @ vprime 
        # (B,nh,T,T) * (B,nh,T,n_emb//nh)
        # (4,12,1024,1024) * (4,12,1024,64)
        # y.shape => (4,12,1024,64)
        # Now we have the context vector per head but lets reassemble them and get the overall context vector
        y = y.transpose(1,2).contiguous().view(x.shape[0],x.shape[1],x.shape[2])
        # (4,12,1024,64) => (4,1024,12,64) => (4,1024,768)
        y = self.c_proj(y)
        # (4,1024,768) * (768* 768) => (4,1024,768)
        return y 
        # you could also return the attention scores if needed.


class HiddenBlock(nn.Module):
    """ 
    Architecture of a single Hidden block
	  Input: Input embeddings <-> (B,T,C) -> (4,1024,768)
	  Output: Context aware embeddings <-> 
	  """
    def __init__(self,config):
        super(HiddenBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadedAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Here x is of shape (4,1024,768).
        # self.ln_1(x) -> Since this a normalization op, there wont be a shape change.
        # self.ln_1(x) -> (4,1024,768).
        x = x + self.attn(self.ln_1(x))
        # Here we make sure that the layer norm is applied first rather than later.
        # This helps us make things stable.
        x=  x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            # wte.weight.shape = (50304,768)
            wpe = nn.Embedding(config.seq_len, config.n_embd),
            # wte.weight.shape = (1024,768)
            h = nn.ModuleList(HiddenBlock(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
            )
        self.transformer = nn.ModuleDict(modules)
        # The thing is that nn.ModuleDict accepts only a dictionary as an object
        # so we either create a dict object inside or define it like I am doing.
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        # lm_head.shape(768,50304)
        self.transformer.wte.weight = self.lm_head.weight
        #Weight tying here
        self.apply(self.init_weights)
        
    def forward(self,x):
		# Here x is of size (4,1024)
        word_embeddings = self.transformer.wte(x)
        # As an embedding layer is nothing but a look table each token is replaced by its respective word embedding
        # word_embeddings.shape -> (4,1024,768)
        # So now we have to get the positional encodings -> Here again we use embeddings which is a look up table.
        # So we have to create a tensor which has all the positions 
        positions = torch.arange(0,x.shape[1], device = x.device)
        # positions.shape -> (4,1024)
        positional_encoding = self.transformer.wpe(positions)
        # positional_encoding.shape -> (4,1024,768)
        x = word_embeddings + positional_encoding
        # (4,1024,768) = (4,1024,768) + (4,1024,768)
        for i in self.transformer.h:
            x = i(x)
        # x.shape -> (4,1024,768)
        x = self.transformer.ln_f(x)
	    # x.shape -> (4,1024,768)
        logits = self.lm_head(x)
	    # (4,1024,768) * (768,50304) => (4,1024,50304)
        return logits
    
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

    @staticmethod
    def count_parameters(model):
        unique_params = set()
        total = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if id(param) not in unique_params:
                    unique_params.add(id(param))
                    total += param.numel()
        return total
    
# class Config:
#     def __init__(self, config_dict):
#         for k, v in config_dict.items():
#             setattr(self, k, v)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)
# config = Config(config)
# sample_input = torch.randint(0, 100, (4,1024)).to(device)
# gpt2 = GPT(config)
# gpt2 = gpt2.to(device)
# print(summary(model = gpt2, input_data=sample_input, depth=4, verbose=0))
# print("###########################")
# print(f"The real nummber of trainable parameters of the model are {GPT.count_parameters(gpt2)}")
# print("###########################")
