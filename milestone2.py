# Install gdown to handle Google Drive file download
from drivingplanner import DrivingPlanner
from logger import Logger
from loader import DrivingDataset
from cmdparser import parser

import numpy as np

import torch

import torch.nn as nn

import pandas as pd
from logging import log
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from train import train, validate
from generate_csv import generate_csv

# Supress dataloader user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

args = parser.parse_args()

# Load the datasets
train_data_dir = "train"
val_data_dir = "val"
test_data_dir = "test_public"

train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]
test_files = [os.path.join(test_data_dir, fn) for fn in sorted([f for f in os.listdir(test_data_dir) if f.endswith(".pkl")], key=lambda fn: int(os.path.splitext(fn)[0]))]

train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)
test_dataset = DrivingDataset(test_files, test=True)

batch_size = int(args.bs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=250, num_workers=2)

# Create save directory for this train
save_dir = os.path.join("models", args.name)
os.makedirs(save_dir, exist_ok=True)
print(f"Created save dir: {save_dir}")

# Create the model
depth_weight = float(args.depth)
sem_weight = float(args.sem)
use_depth_aux = depth_weight > 0.0
use_semantic_aux  = sem_weight > 0.0
model = DrivingPlanner(use_depth_aux=use_depth_aux, use_semantic_aux=use_semantic_aux)

lr = float(args.lr)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Define the loss functions
train_loss_fn = nn.MSELoss()
depth_loss_fn = nn.L1Loss()
sem_loss_fn = nn.CrossEntropyLoss()

logger = Logger()

# Train the model
epochs = int(args.epochs)
best_ade, best_model_dict = train(model, logger, train_loader, val_loader, optimizer, 
    num_epochs=epochs,
    use_depth_aux=use_depth_aux,
    use_semantic_aux=use_semantic_aux, 
    train_loss_fn=train_loss_fn,
    depth_loss_fn=depth_loss_fn,
    sem_loss_fn=sem_loss_fn,
    depth_loss_weight=depth_weight,
    sem_loss_weight=sem_weight
)

print(f"Training's best ADE: {best_ade:.4f}")

# Save the logger metrics
logger.to_csv(save_dir)
logger.plot_metrics(save_dir)

# Validate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ade, fde, mse = validate(model, val_loader, device, train_loss_fn)
print(f"Validation results for model without depth auxiliary loss: ADE: {ade:.4f}, FDE: {fde:.4f}, Traj MSE: {mse:.6f}")

# Save the last and best model
save_path = os.path.join(save_dir, args.name + "_best.pth")
torch.save(best_model_dict, save_path)
save_path = os.path.join(save_dir, args.name + "_last.pth")
torch.save(model.state_dict(), save_path)

# Generate the test CSV for kaggle
best_model = DrivingPlanner()
best_model.load_state_dict(best_model_dict)
generate_csv(model, device, test_loader, save_dir, args.name)