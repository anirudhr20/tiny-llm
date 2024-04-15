### Implementation of the token embedding and position embedding ###
import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
    
    def forward(self, idx):

        B, T = idx.shape 
        tok_emb = self.tok_emb(idx) # (B,T) -> (B,T,C)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb
        return x
        