import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def _optimizer_learning_rates(optimizer):
    learning_rates = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get('name', f'group_{index}')
        learning_rates[f'{group_name}_learning_rate'] = group['lr']
    return learning_rates


def _format_learning_rate_display(optimizer):
    learning_rates = _optimizer_learning_rates(optimizer)
    if len(learning_rates) == 1:
        return f"{optimizer.param_groups[0]['lr']:.6g}"
    return ", ".join(f"{name.replace('_learning_rate', '')}={value:.6g}" for name, value in learning_rates.items())


def _supports_backbone_warmup(model, optimizer):
    has_model_hooks = hasattr(model, 'freeze_backbone') and hasattr(model, 'unfreeze_backbone')
    has_backbone_group = any(group.get('name') == 'backbone' for group in optimizer.param_groups)
    return has_model_hooks and has_backbone_group


def _apply_backbone_warmup_state(model, epoch: int, backbone_warmup_epochs: int):
    if backbone_warmup_epochs <= 0:
        return None

    should_freeze = epoch < backbone_warmup_epochs
    backbone_is_frozen = getattr(model, 'backbone_is_frozen', False)

    if should_freeze and not backbone_is_frozen:
        model.freeze_backbone()
        return 'frozen'
    if not should_freeze and backbone_is_frozen:
        model.unfreeze_backbone()
        return 'unfrozen'

    return None


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
    early_stopping_patience=None,
    early_stopping_min_delta=0.0,
    backbone_warmup_epochs=0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    epoch_history = []
    best_metrics = None
    early_stopping_enabled = early_stopping_patience is not None and early_stopping_patience > 0
    backbone_warmup_enabled = backbone_warmup_epochs > 0
    best_early_stopping_ade = None
    early_stopping_bad_epochs = 0
    stopped_early = False
    best_checkpoint_path = Path(best_checkpoint_path) if best_checkpoint_path is not None else None
    if best_checkpoint_path is not None:
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if backbone_warmup_enabled and not _supports_backbone_warmup(model, optimizer):
        raise ValueError(
            "backbone_warmup_epochs requires a model with freeze_backbone()/unfreeze_backbone() "
            "and an optimizer with a named 'backbone' parameter group."
        )

    for epoch in range(num_epochs):
        backbone_transition = _apply_backbone_warmup_state(model, epoch, backbone_warmup_epochs)
        if backbone_transition == 'frozen':
            print(f"Backbone warmup: freezing ResNet18 backbone for epoch {epoch+1}/{num_epochs}.")
        elif backbone_transition == 'unfrozen':
            print(f"Backbone warmup: unfreezing ResNet18 backbone at epoch {epoch+1}/{num_epochs}.")

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
            'backbone_frozen': bool(getattr(model, 'backbone_is_frozen', False)),
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

        if early_stopping_enabled:
            if (
                best_early_stopping_ade is None
                or epoch_metrics['val_ADE'] < best_early_stopping_ade - early_stopping_min_delta
            ):
                best_early_stopping_ade = epoch_metrics['val_ADE']
                early_stopping_bad_epochs = 0
            else:
                early_stopping_bad_epochs += 1
            epoch_metrics['early_stopping_bad_epochs'] = early_stopping_bad_epochs

        current_learning_rate = optimizer.param_groups[0]['lr']
        learning_rate_metrics = _optimizer_learning_rates(optimizer)
        epoch_history.append(epoch_metrics)
        logger.log(
            step=epoch + 1,
            **epoch_metrics,
            learning_rate=current_learning_rate,
            **learning_rate_metrics,
            best_val_ADE=best_metrics['val_ADE'],
            best_val_ADE_epoch=best_metrics['epoch'],
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_metrics['train_loss']:.4f} | "
            f"Val Loss: {epoch_metrics['val_loss']:.4f} | "
            f"ADE: {epoch_metrics['val_ADE']:.4f} | "
            f"FDE: {epoch_metrics['val_FDE']:.4f} | "
            f"LR: {_format_learning_rate_display(optimizer)} | "
            f"Best ADE: {best_metrics['val_ADE']:.4f} (epoch {best_metrics['epoch']})"
        )

        if early_stopping_enabled and early_stopping_bad_epochs >= early_stopping_patience:
            stopped_early = True
            print(
                f"Early stopping triggered at epoch {epoch+1}: "
                f"validation ADE did not improve by at least {early_stopping_min_delta:.6g} "
                f"for {early_stopping_patience} consecutive epochs."
            )
            logger.log(
                early_stopped=True,
                early_stopping_epoch=epoch + 1,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                best_val_ADE=best_metrics['val_ADE'],
                best_val_ADE_epoch=best_metrics['epoch'],
            )
            break

    final_metrics = epoch_history[-1] if epoch_history else {}
    if best_metrics is None:
        best_metrics = {}

    logger.log(
        training_complete=True,
        best_val_ADE=best_metrics.get('val_ADE'),
        best_val_ADE_epoch=best_metrics.get('epoch'),
        best_checkpoint_path=best_metrics.get('checkpoint_path'),
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopped=stopped_early,
        backbone_warmup_epochs=backbone_warmup_epochs,
    )
    return {
        'epochs_completed': len(epoch_history),
        'final': final_metrics,
        'history': epoch_history,
        'best': best_metrics,
        'final_learning_rate': optimizer.param_groups[0]['lr'],
        'final_learning_rates': _optimizer_learning_rates(optimizer),
        'early_stopping_enabled': early_stopping_enabled,
        'early_stopped': stopped_early,
        'backbone_warmup_epochs': backbone_warmup_epochs,
        'backbone_warmup_enabled': backbone_warmup_enabled,
    }
