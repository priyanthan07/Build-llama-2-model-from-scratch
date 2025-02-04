import torch
import torch.nn as nn
import torch.nn.functional as F # type: ignore
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32                            # number of heads for query
    n_kv_heads: Optional[int] = None             # number of heads for the k and v
    vocab_size: int = -1                         # this will be set when we load the tokenizer
    
    # parameters for feed forward network layers
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None   
    norm_eps: float = 1e-5
    
    # needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
    quant_type: Optional[str] = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta:float = 10000.0):
    """
    According to the paper, RoPE is only applied to even-numbered dimensions.
    
    RoPE:
    - Construct theta parameters as per the formula:
      theta_i = 10000^(-i/dim) for i = [0, 2, 4, ..., dim-2]
      OR
      theta_i = 10000^(-2i/dim) for i = [0,1,2,..., dim/2]
    - Shape = (head_dim / 2)
    """
    
    assert head_dim % 2 == 0, "Dimension must be a even number" 
    
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # theta shape:  (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    # construct the positions. shape : (seq_len)
    m = torch.arange(seq_len, device=device)
    
    # Multiply each theta by each position using the outer product
    # shape: (seq_len) outer_product (head_dim/2) --> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex
    
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_len, H, head_dim) --> (B, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim/2) --> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim/2, 2) --> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (B, seq_len, n_kv_heads, head_dim) --> (B, seq_len, n_kv_heads, 1, head_dim) --> (B, seq_len, n_kv_heads, n_rep, head_dim)
        x = x[:,:,:,None,:].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, seq_len, n_kv_heads, n_rep, head_dim) --> (B, seq_len, n_kv_heads * n_rep, head_dim)
        x = x.reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim)
        return x

class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma parameter
        
    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim)
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (dim) * (B, seq_len, dim) --> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads 
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads       # number of times the key and value are repeated to maatch the headd of the queries
        self.head_dim = args.dim // args.n_heads            # Dimension of each head
        
        # weight metrices for the queries, keys and values
        self.wq = nn.Linear(args.dim, args.n_heads  * self.head_dim, bias=False) 
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # attention output weight matrix
        self.wo = nn.Linear(args.n_heads *self.head_dim, args.dim, bias=False)
        
        # cache
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        """"
            Because of this model is used for inference, we are processing only one token at a time. So the seq_len is 1.
        """
        
        batch_size, seq_len, _ = x.shape     # (B, 1, dim) 
        
        # (B, 1, dim) * (dim, n_heads_q * head_dim) --> (B, 1, n_heads * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) * (dim, n_kv_heads * head_dim) --> (B, 1, n_kv_heads * head_dim)
        xk = self.wk(x) 
        xv = self.wv(x)
        
        # (B, 1, n_head_q * head_dim) --> (B, 1, n_head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, n_kv_heads * head_dim) --> (B, 1, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # RoPE 
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        # kv cache
        # replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv
        
        # Retrieve all the keys and values from the cache
        # (B, seq_len_kv, n_kv_heads, head_dim)
        keys   = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]
        
        # Repeat the heads of the K and V to reach the number of heads of the queries
        keys   = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # (B, 1, n_heads_q, head_dim) --> (B, n_heads_q, 1, head_dim)
        xq     = xq.transpose(1,2)
        # (B, 1, n_kv_heads, head_dim) --> (B, n_kv_heads, 1, head_dim)
        keys   = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        # (B, n_heads_q, 1, head_dim) * (B, n_kv_heads, head_dim, seq_len_kv) --> (B, n_heads_q, 1, seq_len_kv)
        attention_scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim) 
        attention_scores = F.softmax(attention_scores.float(), dim=-1).type_as(xq)
        
        #(B, n_heads_q, 1, seq_len_kv) * (B, n_heads_q, seq_len_kv, head_dim) --> (B, n_heads_q, 1, head_dim)
        out = torch.matmul(attention_scores, values)
        # (B, n_heads_q, 1, head_dim) --> (B, 1, n_heads_q * head_dim)
        out = (out.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(out)   # (B, 1, dim)
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2* hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
            
    def forward(self, x:torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
        

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim =  args.dim // args.n_heads 
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)       # Normalization before the self attention
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)             # Normalization before the feed forward block
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len ,dim ) + (B, seq_len ,dim ) --> (B, seq_len ,dim )
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs ):
        super().__init__()
        
        assert args.vocab_size != -1,   "vocab size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
            
        self.norm = RMSNorm(args.dim, eps= args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len*2, device=self.args.device)
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_length)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token can be processed at a time because of kv cache"
        
        #### This model is good for inference not for training, because in traing we need to process the multiple tokens
        
        h = self.tok_embeddings(tokens)
        
        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # execute all the encode layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output 
    