class GPTConfig:
    def __init__(self, vocab_size, block_size, num_layers=12, num_heads=12, embed_dim=768, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
dataset_path = "../data/input.txt"
model_name = "tinyLLM"
model_path = f"../models/{model_name}.pt"
tokenizer_path = f"../models/{model_name}"

learning_rate = 1e-3
vocab_size = 2048
batch_size = 32 
block_size = 32
num_epochs = 2000
num_heads = 4
num_layers = 4
embed_dim = 256

gpt_config = GPTConfig(
    vocab_size = vocab_size,
    block_size = block_size,
    embed_dim = embed_dim,
    num_heads = num_heads,
    num_layers = num_layers
)
