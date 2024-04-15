import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
from llm.model import TinyLLM
from llm.tokenizer import TinyTokenizer
from config import *

device = "mps" if torch.backends.mps.is_available() else "cpu"
ckpt = torch.load(model_path)
model = TinyLLM(gpt_config.vocab_size, gpt_config.embed_dim, gpt_config.num_heads, gpt_config.num_layers, gpt_config.block_size)
model.load_state_dict(ckpt)
model = model.to(device)
tokenizer = TinyTokenizer(train=False, model_path = f"{tokenizer_path}.model")

def generate(model, idx, max_new_tokens):
    with torch.no_grad():
        model.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # print(idx_cond)
            logits, loss  = model(idx_cond)
            logits = logits[:, -1, :] # (B,T,V) -> (B,V)
            # print(logits.shape)
            # print(logits)
            probs = F.softmax(logits, dim=-1)
            # print(probs)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    output = tokenizer.decode(idx[0].tolist())
    return output

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
    print(generate(model, input_ids.to(device), 100))

