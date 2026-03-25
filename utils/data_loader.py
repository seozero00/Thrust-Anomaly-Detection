import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TimeDataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.csv_files = sorted([f for f in os.listdir(root_folder) if f.startswith("Step") and f.endswith(".csv")])

        if not self.csv_files:
            raise FileNotFoundError(f"No CSV files found in {root_folder}")

        self.variables = ["P", "Q", "R", "p_dot", "q_dot", "r_dot",
                        "U", "V", "W", "u_dot", "v_dot", "w_dot"] 

    def __len__(self):
        return len(self.csv_files)  

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_folder, self.csv_files[idx])
        df = pd.read_csv(file_path)

        features = torch.tensor(df[self.variables].values, dtype=torch.float32).T  
        return features  

def build_dataloader(root_folder, batch_size=32):
    dataset = TimeDataset(root_folder)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader
