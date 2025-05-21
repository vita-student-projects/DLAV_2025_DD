import torch 
import torch.nn as nn
import numpy as np

def train(model, logger, train_loader, val_loader, optimizer, num_epochs=50, criterion=nn.MSELoss(), scheduler=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_model_dict = None
    best_ade = float('inf')

    for epoch in range(num_epochs):
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
            if scheduler:
                scheduler.step(loss)

            if idx % 10 == 0:
                ade = torch.norm(pred_future[:, :, :2] - future[:, :, :2], p=2, dim=-1).mean()
                fde = torch.norm(pred_future[:, -1, :2] - future[:, -1, :2], p=2, dim=-1).mean()
                metrics = {
                    'loss': loss.item(),
                    'ADE': ade.item(),
                    'FDE': fde.item()
                }
                logger.log(step=epoch * len(train_loader) + idx, **metrics)
            train_loss += loss.item()

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

        # Save the best model
        if np.mean(ade_all) < best_ade:
            best_ade = np.mean(ade_all)
            best_model_dict = model.state_dict()

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | ADE: {np.mean(ade_all):.4f} | FDE: {np.mean(fde_all):.4f}')
    
    return best_ade, best_model_dict