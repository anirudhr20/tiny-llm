### Training script for TinyLLM model ###
import sys
sys.path.append('../')
from llm.model import TinyLLM
from llm.tokenizer import TinyTokenizer
from config import *
from utils import read_data, train_test_split_data, get_data_batch
import torch 


texts = read_data(dataset_path)

    
# tokenizer = TinyTokenizer(train=True)
tokenizer = TinyTokenizer(train=False, model_path = f"{tokenizer_path}.model")
# tokenizer.train(texts, vocab_size, verbose=True, model_path=f"{tokenizer_path}.model", vocab_path=f"{tokenizer_path}.vocab")
print(f"Tokenizer trained with vocab size {len(tokenizer.vocab)} and merges {len(tokenizer.merges)}")


data = torch.tensor(tokenizer.encode(texts), dtype=torch.long)
train_data, test_data = train_test_split_data(data, split_ratio=0.9)


device = "mps" if torch.backends.mps.is_available() else "cpu"

model = TinyLLM(gpt_config.vocab_size, gpt_config.embed_dim, gpt_config.num_heads, gpt_config.num_layers, gpt_config.block_size).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model.train()
for iter in range(num_epochs):
    xb, yb = get_data_batch(train_data, block_size, batch_size)
    
    logits, loss = model(xb.to(device), yb.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 50 == 0:
        print(f"Iter: {iter}, Loss: {loss.item()}")

torch.save(model.state_dict(), f"{model_path}")

