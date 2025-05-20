from re import A
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def train_one_epoch(epoch, model, logger, train_loader, optimizer, device, criterion):
    # Training
    model.train()
    train_loss = 0
    for idx, batch in enumerate(train_loader):
        camera = batch['camera'].to(device)
        history = batch['history'].to(device)
        future = batch['future'].to(device)

        optimizer.zero_grad()
        pred_future, _, _ = model(camera, history)
        loss = criterion(pred_future[..., :2], future[..., :2])
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            logger.log(step=epoch * len(train_loader) + idx, loss=loss.item())
        train_loss += loss.item()
            
    return train_loss / len(train_loader)
            
def validate(model, val_loader, device, criterion):
    # Validation
    model.eval()
    val_loss, ade_all, fde_all = 0, [], []
    with torch.no_grad():
        for batch in val_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            future = batch['future'].to(device)

            pred_future, _, _ = model(camera, history)
            loss = criterion(pred_future, future)
            ADE = torch.norm(pred_future[:, :, :2] - future[:, :, :2], p=2, dim=-1).mean()
            FDE = torch.norm(pred_future[:, -1, :2] - future[:, -1, :2], p=2, dim=-1).mean()
            ade_all.append(ADE.item())
            fde_all.append(FDE.item())
            val_loss += loss.item()

    return val_loss / len(val_loader), np.mean(ade_all), np.mean(fde_all)

def train(model, logger, train_loader, val_loader, optimizer, num_epochs=50, criterion=nn.MSELoss()):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    best_ade = 99
    best_model = None

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            epoch,
            model,
            logger, 
            train_loader, 
            optimizer, 
            device, 
            criterion
        )

        mse, ade, fde = validate(model, val_loader, device, criterion)

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {mse:.4f} | ADE: {ade:.4f} | FDE: {fde:.4f}')

        if ade < best_ade:
            best_model = model.state_dict()
            best_ade = ade

    return best_ade, best_model