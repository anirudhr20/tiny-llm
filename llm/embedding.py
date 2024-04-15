### Implementation of the token embedding and position embedding ###
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size) -> None:
        super().__init__(vocab_size, embed_size)

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, block_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.pe = nn.Embedding(block_size, embedding_size)
        
    def forward(self, x):
        return self.pe(torch.arange(self.block_size).to(x.device))

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size) -> None:
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, embed_size)
        self.pos_emb = PositionalEmbedding(embed_size, block_size)
    
    def forward(self, x):
        return self.tok_emb(x) + self.pos_emb(x)
        