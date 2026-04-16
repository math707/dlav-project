import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    logger,
    num_epochs=50,
    scheduler=None,
    scheduler_metric='val_ADE',
    best_checkpoint_path=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    epoch_history = []
    best_metrics = None
    best_checkpoint_path = Path(best_checkpoint_path) if best_checkpoint_path is not None else None
    if best_checkpoint_path is not None:
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for idx, batch in enumerate(train_loader):
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            future = batch['future'].to(device)

            optimizer.zero_grad()
            pred_future = model(camera, history)
            loss = criterion(pred_future[..., :2], future[..., :2])
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                logger.log(step=epoch * len(train_loader) + idx, loss=loss.item())
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, ade_all, fde_all = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                camera = batch['camera'].to(device)
                history = batch['history'].to(device)
                future = batch['future'].to(device)

                pred_future = model(camera, history)
                loss = criterion(pred_future, future)
                ADE = torch.norm(pred_future[:, :, :2] - future[:, :, :2], p=2, dim=-1).mean()
                FDE = torch.norm(pred_future[:, -1, :2] - future[:, -1, :2], p=2, dim=-1).mean()
                ade_all.append(ADE.item())
                fde_all.append(FDE.item())
                val_loss += loss.item()

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_ADE': float(np.mean(ade_all)),
            'val_FDE': float(np.mean(fde_all)),
        }

        if best_metrics is None or epoch_metrics['val_ADE'] < best_metrics['val_ADE']:
            best_metrics = dict(epoch_metrics)
            if best_checkpoint_path is not None:
                torch.save(model.state_dict(), best_checkpoint_path)
                best_metrics['checkpoint_path'] = str(best_checkpoint_path)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_metrics[scheduler_metric])
            else:
                scheduler.step()

        current_learning_rate = optimizer.param_groups[0]['lr']
        epoch_history.append(epoch_metrics)
        logger.log(
            step=epoch + 1,
            **epoch_metrics,
            learning_rate=current_learning_rate,
            best_val_ADE=best_metrics['val_ADE'],
            best_val_ADE_epoch=best_metrics['epoch'],
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_metrics['train_loss']:.4f} | "
            f"Val Loss: {epoch_metrics['val_loss']:.4f} | "
            f"ADE: {epoch_metrics['val_ADE']:.4f} | "
            f"FDE: {epoch_metrics['val_FDE']:.4f} | "
            f"LR: {current_learning_rate:.6g} | "
            f"Best ADE: {best_metrics['val_ADE']:.4f} (epoch {best_metrics['epoch']})"
        )

    final_metrics = epoch_history[-1] if epoch_history else {}
    if best_metrics is None:
        best_metrics = {}

    logger.log(
        training_complete=True,
        best_val_ADE=best_metrics.get('val_ADE'),
        best_val_ADE_epoch=best_metrics.get('epoch'),
        best_checkpoint_path=best_metrics.get('checkpoint_path'),
    )
    return {
        'epochs_completed': len(epoch_history),
        'final': final_metrics,
        'history': epoch_history,
        'best': best_metrics,
        'final_learning_rate': optimizer.param_groups[0]['lr'],
    }
