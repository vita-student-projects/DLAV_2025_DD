# Install gdown to handle Google Drive file download
import pickle

from drivingplanner import DrivingPlanner2
from logger import Logger
from loader import DrivingDataset
from cmdparser import parser

import numpy as np
import random

import torch
import pickle

import torch.nn as nn

import pandas as pd
from logging import log
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F

# Supress dataloader user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

k = 4
# load the data
data = []
for i in random.choices(np.arange(1000), k=k):
    with open(f"train/{i}.pkl", "rb") as f:
        data.append(pickle.load(f))

def compute_depth_loss(pred, gt):
    # pred, gt: (B, H, W, 1)
    pred = pred.squeeze(-1)
    gt = gt.squeeze(-1)

    l1_loss = F.l1_loss(pred, gt)

    # SSIM Loss (simplified)
    def ssim(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

    # SSIM expects (B, 1, H, W)
    ssim_loss = ssim(pred.unsqueeze(1), gt.unsqueeze(1)).mean()

    # Smoothness loss (optional)
    grad_pred_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
    grad_pred_y = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])
    smooth_loss = grad_pred_x.mean() + grad_pred_y.mean()

    total_loss = l1_loss + 0.85 * ssim_loss + 0.1 * smooth_loss
    return total_loss

class ADELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction  # 'mean' or 'sum' or 'none'

    def forward(self, pred, target):
        """
        pred: (B, T, 2) predicted future positions
        target: (B, T, 2) ground truth future positions
        """
        displacement = torch.norm(pred - target, dim=2)  # (B, T)
        if self.reduction == 'mean':
            return displacement.mean()
        elif self.reduction == 'sum':
            return displacement.sum()
        else:
            return displacement  # (B, T)

def train_one_epoch(epoch, model, train_loader, optimizer, device, lambda_depth=0.1, use_depth_aux=False):
    model.train()
    train_loss = 0.0
    for idx, batch in enumerate(train_loader):
        cam, hist, fut, dep, sem = [batch[k].to(device) for k in ['camera', 'history', 'future', 'depth', 'semantic']]
        optimizer.zero_grad()
        fut_pred, dep_pred, sem_pred = model(cam, hist, dep)

        traj_loss = F.mse_loss(fut_pred, fut)
        loss = traj_loss
        if use_depth_aux:
            loss += lambda_depth * compute_depth_loss(dep_pred, dep)
            loss += 0.2 * F.cross_entropy(sem_pred, sem.long(), reduction='mean')
        
        if idx % 10 == 0:
            ADE = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], p=2, dim=-1).mean()
            FDE = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], p=2, dim=-1).mean()

            metrics = {
                'loss': loss.item(),
                'ADE': ADE.item(),
                'FDE': FDE.item()
            }
            logger.log(step=epoch * len(train_loader) + idx, **metrics)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, device):
    model.eval()
    total_ade, total_fde, total_mse = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            cam = batch['camera'].to(device)
            hist = batch['history'].to(device)
            fut = batch['future'].to(device)
            dep = batch['depth'].to(device)

            fut_pred, _, _ = model(cam, hist, dep)

            B, T, _ = fut.shape
            count += B

            ade = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], dim=2).mean(dim=1).sum()
            fde = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], dim=1).sum()
            mse = F.mse_loss(fut_pred, fut, reduction='sum')

            total_ade += ade.item()
            total_fde += fde.item()
            total_mse += mse.item()

    ade_avg = total_ade / count
    fde_avg = total_fde / count
    mse_avg = total_mse / (count * T * 3)

    return ade_avg, fde_avg, mse_avg

def train(model, train_loader, val_loader, optimizer, num_epochs=50, lambda_depth=0.1, use_depth_aux=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, device, lambda_depth, use_depth_aux)
        ade, fde, mse = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Validation - ADE: {ade:.4f}, FDE: {fde:.4f}, Traj MSE: {mse:.6f}")


# Read the save path from the cmd line arguments
args = parser.parse_args()

loss = args.loss
if loss == "mse":
    criterion = nn.MSELoss()
elif loss == "ade":
    criterion = ADELoss()
elif loss == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
elif loss == "bce":
    criterion = nn.BCELossWithLogits()
else:
    criterion = nn.MSELoss()

train_data_dir = "train"
val_data_dir = "val"

train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)

batch_size = int(args.bs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)


save_dir = os.path.join("models", args.name)
os.makedirs(save_dir, exist_ok=True)
print(f"Created save dir: {save_dir}")

use_depth_aux = args.depth
model = DrivingPlanner2(use_depth_aux=use_depth_aux)


lr = float(args.lr)
optimizer = optim.Adam(model.parameters(), lr=lr)

logger = Logger()

epochs = int(args.epochs)
ld = float(args.ld)
train(model, train_loader, val_loader, optimizer, num_epochs=epochs,use_depth_aux=use_depth_aux, lambda_depth=ld)

# Save the logger metrics
logger.to_csv(save_dir)
logger.plot_metrics(save_dir)

# Validate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ade, fde, mse = validate(model, val_loader, device)
print(f"Validation results for model without depth auxiliary loss: ADE: {ade:.4f}, FDE: {fde:.4f}, Traj MSE: {mse:.6f}")


# save the model
save_path = os.path.join(save_dir, args.name + ".pth")
torch.save(model.state_dict(), save_path)
print("Model saved as", save_path) # Create a folder with the name to put the results

test_data_dir = "test_public"
test_files = [os.path.join(test_data_dir, fn) for fn in sorted([f for f in os.listdir(test_data_dir) if f.endswith(".pkl")], key=lambda fn: int(os.path.splitext(fn)[0]))]
test_dataset = DrivingDataset(test_files, test=True)
test_loader = DataLoader(test_dataset, batch_size=250, num_workers=2)
model.eval()
all_plans = []
with torch.no_grad():
    for batch in test_loader:
        camera = batch['camera'].to(device)
        history = batch['history'].to(device)
        dep = batch['depth'].to(device)

        pred_future, _ = model(camera, history, dep)
        all_plans.append(pred_future.cpu().numpy()[..., :2])
all_plans = np.concatenate(all_plans, axis=0)

# Now save the plans as a csv file
pred_xy = all_plans[..., :2]  # shape: (total_samples, T, 2)

# Flatten to (total_samples, T*2)
total_samples, T, D = pred_xy.shape
pred_xy_flat = pred_xy.reshape(total_samples, T * D)

# Build a DataFrame with an ID column
ids = np.arange(total_samples)
df_xy = pd.DataFrame(pred_xy_flat)
df_xy.insert(0, "id", ids)

# Column names: id, x_1, y_1, x_2, y_2, ..., x_T, y_T
new_col_names = ["id"]
for t in range(1, T + 1):
    new_col_names.append(f"x_{t}")
    new_col_names.append(f"y_{t}")
df_xy.columns = new_col_names

# Save to CSV
df_xy.to_csv("submission_phase2.csv", index=False)

print(f"Shape of df_xy: {df_xy.shape}")