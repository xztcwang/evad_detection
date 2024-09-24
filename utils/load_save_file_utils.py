import json
import os
import random
import torch
import numpy as np

def save_data(data_file_path, file_name, data):
    data_file = os.path.join(data_file_path, file_name)
    # write the data to a json file in the save folder
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"data written into {data_file}")

def load_data(data_file_path, file_name):
    data_file = os.path.join(data_file_path, file_name)
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"data loaded from {data_file}")
    return data

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True