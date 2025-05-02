import pickle

from drivingplanner import DrivingPlanner
from logger import Logger
from loader import DrivingDataset
from cmdparser import parser

import numpy as np

import torch
import pickle

from cmdparser import parser

import pandas as pd
from logging import log
from torch.utils.data import DataLoader
import os

parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
parser.add_argument("--save_dir", type=str, default="tests", help="Directory to save the predictions")

args = parser.parse_args()

model_path = args.model
model = DrivingPlanner(history_encoder=args.hencoder, ego_encoding=args.ego)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

train_data_dir = "train"
val_data_dir = "val"

train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)

batch_size = int(args.bs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

val_batch_zero = next(iter(val_loader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
camera = val_batch_zero['camera'].to(device)
history = val_batch_zero['history'].to(device)
future = val_batch_zero['future'].to(device)

model.eval()
with torch.no_grad():
    pred_future = model(camera, history)

camera = camera.cpu().numpy()
history = history.cpu().numpy()
future = future.cpu().numpy()
pred_future = pred_future.cpu().numpy()

with open(f"test_public/0.pkl", "rb") as f:
    data = pickle.load(f)
# Note the absence of sdc_future_feature

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

        pred_future = model(camera, history)
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
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, args.name + ".csv")
df_xy.to_csv(save_path, index=False)
print(f"Predictions saved to {save_path}")