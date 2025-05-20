import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms as v2
import numpy as np

class DrivingDataset(Dataset):

    def flipdata(camera, history, future):
        camera_flipped = v2.RandomHorizontalFlip(p=1)(camera)

        history_flipped = history
        history_flipped[:,1] = -history_flipped[:,1]
        history_flipped[:,2] = (history_flipped[:, 2] + np.pi) % (2 * np.pi) - np.pi
        
        future_flipped = future
        future_flipped[:,1] = -future_flipped[:,1]
        future_flipped[:,2] = (future_flipped[:, 2] + np.pi) % (2 * np.pi) - np.pi
        
        return camera_flipped, history_flipped, future_flipped

    def __init__(self, file_list, test=False, flip=False):
        self.samples = file_list
        self.test = test
        self.flip = flip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load pickle file
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        # Convert numpy arrays to tensors
        camera = torch.FloatTensor(data['camera']).permute(2, 0, 1)
        history = torch.FloatTensor(data['sdc_history_feature'])
        if not self.test:
          future = torch.FloatTensor(data['sdc_future_feature'])
          if self.flip:
            camera, history, future = DrivingDataset.flipdata(camera, history, future)
          return {
            'camera': camera,
            'history': history,
            'future': future
          }
        else:
          return {
            'camera': camera,
            'history': history
          }

