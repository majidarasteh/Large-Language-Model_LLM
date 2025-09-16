
# implemented_modules.py

# Read the file
import os  
import urllib.request
import math

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

"""
     Chapter 03- Preparation of text data.

    * GPTDatasetV1: A PyTorch Dataset that takes a string of text and chunks it into overlapping input-target pairs for the next-token prediction task. It's the foundation for creating a training dataset.

    * create_dataloader_v1: A convenience function that creates a PyTorch DataLoader from the GPTDatasetV1. This handles batching, shuffling, and iterating over the dataset, which is crucial for efficient training.
"""

# GPTDatasetV1: Takes raw text and splits it into pairs of input_chunk + target_chunk.
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)    

        for i in range(0, len(token_ids) - max_length, stride):     
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):    
        return len(self.input_ids)

    def __getitem__(self, idx):         
        return self.input_ids[idx], self.target_ids[idx]


# Data loader = Efficient text-to-batches converter.
def create_dataloader_v1(txt,            # The input text data to process.
                         batch_size=4,   # Number of input-target pairs per batch (4 sequences processed simultaneously).
                         max_length=256, # Maximum length (in tokens) for each input sequence.
                         stride=128,     # How many tokens the sliding window moves forward between sequences.
                         shuffle=True,   # Whether to randomize the order of sequences before batching.
                         drop_last=True, # If True, discards incomplete batches at the end.
                         num_workers=0   #Number of CPU cores for parallel data loading.
                        ):
    tokenizer = tiktoken.get_encoding('gpt2')                        
    dataset = GPTDatasetV1(txt,        # Your text
                           tokenizer,  # Your tokenizer object
                           max_length, # Number of tokens
                           stride      # Tokens to slide window
                          ) 
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,     
        num_workers=num_workers     
    )

    return dataloader


"""
   Chapter 07- Attention mechanisms.

   * MultiHeadAttention: The heart of the transformer architecture. Your implementation is efficient and correct:
       - It uses a single linear layer per type (Q, K, V) and then splits the resulting large tensor into multiple heads using .view() and .transpose(), which is computationally efficient.
       - It correctly implements causal masking using a buffer to prevent the model from attending to future tokens.
       - It includes an optional output projection layer (out_proj) and dropout for regularization.
"""

# Implementing multi-head attention with weight splits
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Ensures the output dimension can be split evenly among heads
        # Example: If d_out=8 and num_heads=4, each head gets 2 dimensions
        assert (d_out % num_heads == 0), \
            'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads    
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)# Single large projection matrices (more efficient than separate ones per head)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)   # Optional layer to mix information from different heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)         # [b, num_tokens, d_out]
        queries = self.W_query(x)    # [b, num_tokens, d_out]
        values = self.W_value(x)     # [b, num_tokens, d_out]

        """
            Reshape for Multiple Heads
            Reshapes [b, T, d_out] → [b, T, h, d_h] where d_out = h * d_h
            Example: [2, 6, 8] → [2, 6, 4, 2] (4 heads, 2 dims each)
        """
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(                                             
            b, num_tokens, self.num_heads, self.head_dim                    
        )                                                                   

        """
            Transpose for Batch Computation
            Rearranges to [batch, heads, tokens, dims_per_head]
            Allows parallel computation across heads
        """
        keys = keys.transpose(1, 2)          # [b, h, T, d_h]
        queries = queries.transpose(1, 2)    # [b, h, T, d_h]
        values = values.transpose(1, 2)      # [b, h, T, d_h]

        """
            Compute Attention Scores
            Batched matrix multiplication across all heads
            Computes all attention scores in parallel
        """
        attn_scores = queries @ keys.transpose(2, 3)  

        """
            Apply Causal Mask
            Uses pre-computed triangular mask
            Blocks future tokens for autoregressive generation
        """
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        
        # Softmax: Standard scaled softmax attention
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Dropout for regularization
        attn_weights = self.dropout(attn_weights)

        """
            Apply Attention to Values
            Weighted sum of values
            ranspose back to [batch, tokens, heads, dims]
        """
        context_vec = (attn_weights @ values).transpose(1, 2)   # [b, T, h, d_h]

        """
            Combine Heads
            Flatten heads: [b, T, h, d_h] → [b, T, h*d_h] = [b, T, d_out]
            Example: [2, 6, 4, 2] → [2, 6, 8]
        """
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )

        """
            Output Projection
            Optional linear transformation
            Helps mix information across heads
        """
        context_vec = self.out_proj(context_vec)    #11
        return context_vec


"""
    Chapter 09- Implementing GPT Model        
"""


""" Layer Normalization Implementation

    * LayerNorm: Implements layer normalization, which stabilizes training by normalizing the inputs across the embedding dimension. It includes learnable scale and shift parameters.
"""
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift



"""
      GELU Activation Function
    * GELU: Implements the Gaussian Error Linear Units activation function, which is a smooth and performant alternative to ReLU used in modern transformers.
"""

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))

"""
    FeedForward Network
    * FeedForward: The position-wise feed-forward network within each transformer block. It expands the dimension by a factor of 4 and then contracts it back, adding non-linearity and processing power to the model.
"""

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)



"""
    Transformer Block
        * TransformerBlock: A single block of the GPT architecture. It combines:
        - Multi-head self-attention
        - A feed-forward network
        - Layer normalization applied before each sub-layer (Pre-LN), which is standard and improves training stability.
        - Residual (shortcut) connections around each sub-layer.
        - Dropout for regularization.
"""

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'], 
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # Self-attention with shortcut connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        # Feed-forward with shortcut connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x



"""
    GPT Model Architecture
    * GPTModel: The full model architecture. It combines:
        - Token Embeddings: To convert input token IDs into vectors.
        - Positional Embeddings: To give the model information about the order of tokens.
        - A dropout layer on the combined embeddings.
        - A stack of n_layers TransformerBlocks.
        - A final layer normalization.
        - An output head that projects the final hidden states back to the vocabulary size to generate logits for the next token.
"""

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        # Token embeddings
        tok_embeds = self.tok_emb(in_idx)
        
        # Position embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Combine embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # Transformer blocks
        x = self.trf_blocks(x)
        
        # Final normalization and output
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits



"""
    * generate_text_simple: 
     - A function that performs greedy decoding. 
     - It takes an initial prompt and autoregressively generates new tokens by always choosing the token with the highest probability at each step. 
     - This is a simple but effective way to generate text.
     
"""
# Text Generation Function
def generate_text_simple(model, idx, max_new_tokens, context_size):

    # Loop for Token Generation
    # Generates one token per iteration until max_new_tokens are created.
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx[:, -context_size:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on last time step
        logits = logits[:, -1, :]
        
        # Get probabilities
        probas = torch.softmax(logits, dim=-1)
        
        # Sample next token (greedy)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

"""
    * The if __name__ == "__main__": block is excellent for testing. It:
        - Defines the configuration for a small GPT-2-like model (124M parameters).
        - Initializes the model and sets it to evaluation mode.
        - Tokenizes a start context ("Hello, I am").
        - Generates 10 new tokens using the generate_text_simple function.
        -Prints the input, the generated token IDs, and the decoded output text.
"""

if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)




    