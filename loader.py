import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import pickle
import torch

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class DrivingDataset(Dataset):
    def __init__(self, file_list, test=False):
        self.samples = file_list
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        camera = torch.FloatTensor(data['camera']).permute(2, 0, 1) / 255.0
        history = torch.FloatTensor(data['sdc_history_feature'])

        depth = torch.FloatTensor(data['depth'])
        sem  = torch.FloatTensor(data['semantic_label'])

        if not self.test:
            future = torch.FloatTensor(data['sdc_future_feature'])
            return {
                'camera': camera,
                'history': history,
                'depth': depth,
                'future': future,
                'semantic': sem
            }
        else:
            return {
                'camera': camera,
                'history': history,
                'depth': depth,
                'semantic': sem
            }