### Implementation of the decoder block - Self Attention and MultiHead Attention ###
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_size, dropout_val = 0.1) -> None:
        super().__init__()
        self.head_size = head_size
        self.embed_size = embed_size
        
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.dropout = nn.Dropout(dropout_val)
    
    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        
        wei = q@k.transpose(2,1) # (B, T, T)
        wei = wei / self.head_size**-0.5
        tril = torch.tril(torch.ones(T,T)).to(x.device)
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, H)
        out = wei@v # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_size, attn_dropout_rate = 0.1) -> None:
        super().__init__()
        self.attn_heads = nn.ModuleList([SelfAttention(embed_size=embed_dim, head_size=head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads*head_size, embed_dim)
        self.dropout = nn.Dropout(attn_dropout_rate)
    def forward(self, x):
        B, T, C = x.shape
        out = torch.cat([attn_head(x) for attn_head in self.attn_heads], dim = -1)
        out = self.dropout(self.projection(out))
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout_rate = 0.1) -> None:
        super().__init__()
        head_size = embed_dim // num_heads
        self.mha = MultiHeadAttention(num_heads, embed_dim, head_size, attn_dropout_rate)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(attn_dropout_rate)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, T, C =  x.shape 
        
        out = self.mha(x)

        out = self.ln1(out + x)

        out = self.ffn(out)
        return out