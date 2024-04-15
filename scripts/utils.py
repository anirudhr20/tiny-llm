import torch
def read_data(filepath:str)->str:
    """This function is used to read the training data

    Args:
        filepath (str): The location of the filepath

    Returns:
        str: The content of the given file.
    """
    with open(filepath, "r") as f:
        data = f.read()
    return data

def train_test_split_data(data, split_ratio = 0.8):
    """This function is used to split the data into training and testing data

    Args:
        data (str): The data that needs to be split
        split_ratio (float, optional): The ratio of splitting. Defaults to 0.8.

    Returns:
        Tuple[str,str]: The training and testing data
    """
    
    split_idx = int(len(data)*split_ratio)
    return data[:split_idx], data[split_idx:]

def get_data_batch(data, block_size, batch_size):
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y 
