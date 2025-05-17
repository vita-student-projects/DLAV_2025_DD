import torch
import numpy as np
import os
import pandas as pd

def generate_csv(model, device, test_loader, save_dir, name):
    model.eval()
    all_plans = []
    with torch.no_grad():
        for batch in test_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)

            pred_future, _, _ = model(camera, history)
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
    save_path = os.path.join(save_dir, name+'.csb')
    df_xy.to_csv(save_path, index=False)

    print(f"Shape of df_xy: {df_xy.shape}")