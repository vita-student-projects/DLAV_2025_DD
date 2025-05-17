from re import A
import torch
import torch.nn.functional as F
import torch.nn as nn

def train_one_epoch(epoch, model, logger, train_loader, optimizer, device, train_loss_fn, depth_loss_fn, 
                    sem_loss_fn, use_depth_aux=False, use_semantic_aux=False,
                    depth_loss_weight=0.0, sem_loss_weight=0.0):
    model.train()
    train_loss = 0.0
    depth_loss = 0.0
    sem_loss = 0.0
    for idx, batch in enumerate(train_loader):
        cam, hist, fut, dep, sem = [batch[k].to(device) for k in ['camera', 'history', 'future', 'depth', 'semantic']]
        optimizer.zero_grad()
        fut_pred, dep_pred, sem_pred = model(cam, hist)
        traj_loss = train_loss_fn(fut_pred, fut)
        loss = traj_loss

        if use_depth_aux:
            d_loss = depth_loss_fn(dep_pred, dep)
            loss += depth_loss_weight * d_loss
            depth_loss += d_loss
        if use_semantic_aux:
            s_loss = sem_loss_fn(sem_pred, sem.long())
            loss += sem_loss_weight * s_loss
            sem_loss += s_loss

        if idx % 10 == 0:
            ADE = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], p=2, dim=-1).mean()
            FDE = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], p=2, dim=-1).mean()

            metrics = {
                'loss': loss.item(),
                'ADE': ADE.item(),
                'FDE': FDE.item()
            }

            if use_depth_aux:
                metrics['depth_loss'] = d_loss.item()
            if use_semantic_aux:
                metrics['semantic_loss'] = s_loss.item()

            logger.log(step=epoch * len(train_loader) + idx, **metrics)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_depth_loss = depth_loss / len(train_loader)
    avg_sem_loss = sem_loss / len(train_loader)
    return avg_train_loss, avg_depth_loss, avg_sem_loss

def validate(model, val_loader, device, loss_fn):
    model.eval()
    total_ade, total_fde, total_mse = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            cam = batch['camera'].to(device)
            hist = batch['history'].to(device)
            fut = batch['future'].to(device)
            dep = batch['depth'].to(device)

            fut_pred, _, _ = model(cam, hist)

            B, T, _ = fut.shape
            count += B

            ade = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], dim=2).mean(dim=1).sum()
            fde = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], dim=1).sum()
            mse = F.mse_loss(fut_pred, fut, reduction='sum')
            ade_loss = loss_fn(fut_pred, fut)

            total_ade += ade.item()
            total_fde += fde.item()
            total_mse += mse.item()

    ade_avg = total_ade / count
    fde_avg = total_fde / count
    mse_avg = total_mse / (count * T * 3)

    return ade_avg, fde_avg, mse_avg

def train(model, logger, train_loader, val_loader, optimizer, num_epochs=50, use_depth_aux=False, use_semantic_aux=False,
          train_loss_fn=nn.MSELoss(), depth_loss_fn=nn.L1Loss(), sem_loss_fn=nn.CrossEntropyLoss(),
          depth_loss_weight=0.0, sem_loss_weight=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    best_ade = 99
    best_model = None

    for epoch in range(num_epochs):
        train_loss, depth_loss, sem_loss = train_one_epoch(
            epoch,
            model,
            logger, 
            train_loader, 
            optimizer, 
            device, 
            train_loss_fn, 
            depth_loss_fn, 
            sem_loss_fn, 
            use_depth_aux, 
            use_semantic_aux,
            depth_loss_weight=depth_loss_weight,
            sem_loss_weight=sem_loss_weight
        )

        ade, fde, mse = validate(model, val_loader, device, train_loss_fn)

        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Validation - ADE: {ade:.4f}, FDE: {fde:.4f}, Traj MSE: {mse:.6f}, Depth Loss: {depth_loss:.4f}, Semantic Loss: {sem_loss:.4f}")

        if ade < best_ade:
            best_model = model.state_dict()
            best_ade = ade

    return best_ade, best_model