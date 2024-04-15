### Implementing a tiny version of GPT model ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import TransformerEmbedding
from .decoder import DecoderBlock

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, attn_dropout_rate = 0.1) -> None:
        super().__init__()
        
        # self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # self.positional_embedding = nn.Embedding(block_size, embed_dim)
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, block_size)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(embed_dim, num_heads, attn_dropout_rate) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape 
        # tok_emb = self.token_embedding(idx) # (B,T) -> (B,T,C)
        # pos_emb = self.positional_embedding(torch.arange(T, device=idx.device)) # (T,C)
        # x = tok_emb + pos_emb
        # print(x.shape)
        x = self.embedding(idx) # (B,T) -> (B,T,C)
        x = self.decoder_blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B,T,C) -> (B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
        
        