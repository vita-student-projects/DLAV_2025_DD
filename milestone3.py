# Install gdown to handle Google Drive file download
from drivingplanner import DrivingPlanner
from logger import Logger
from loader import DrivingDataset
from cmdparser import parser
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

import torch

import torch.nn as nn

import pandas as pd
from logging import log
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from train import train
from generate_csv import generate_csv

# Supress dataloader user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

args = parser.parse_args()

# Load the datasets
train_data_dir = "train"
real_data_dir = "val_real"
test_data_dir = "test_public_real"

train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_real_files = [os.path.join(real_data_dir, f) for f in os.listdir(real_data_dir) if f.endswith('.pkl')]
train_files_mixed = train_files + val_real_files[:500]
val_files = val_real_files[500:]
test_files = [os.path.join(test_data_dir, fn) for fn in sorted([f for f in os.listdir(test_data_dir) if f.endswith(".pkl")], key=lambda fn: int(os.path.splitext(fn)[0]))]

train_dataset = DrivingDataset(train_files_mixed)
train_dataset_flipped = DrivingDataset(train_files_mixed, flip=True)
train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flipped])
val_dataset = DrivingDataset(val_files)
test_dataset = DrivingDataset(test_files, test=True)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=250, num_workers=2)

# Create save directory for this train
save_dir = os.path.join("models", args.name)
os.makedirs(save_dir, exist_ok=True)
print(f"Created save dir: {save_dir}")

# Create the model
model = DrivingPlanner(use_depth_aux=False, use_semantic_aux=False)

lr = float(args.lr)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Define the loss functions
criterion = nn.MSELoss()

logger = Logger()

# Train the model
epochs = int(args.epochs)
best_ade, best_model_dict = train(
    model, 
    logger, 
    train_loader, 
    val_loader, 
    optimizer, 
    num_epochs=epochs,
    criterion=criterion,
)

print(f"Training's best ADE: {best_ade:.4f}")

# Save the logger metrics
logger.to_csv(save_dir)
logger.plot_metrics(save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save the last and best model
best_model_path = os.path.join(save_dir, args.name + "_best.pth")
torch.save(best_model_dict, best_model_path)
last_model_path = os.path.join(save_dir, args.name + "_last.pth")
torch.save(model.state_dict(), last_model_path)

# Generate the test CSV for kaggle
best_model = DrivingPlanner(use_depth_aux=False, use_semantic_aux=False)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
generate_csv(best_model, device, test_loader, save_dir)