"""Phase 3 training helpers for sim-to-real trajectory prediction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


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


def _get_future_xy(future):
    if future.shape[-1] < 2:
        raise ValueError(f'Expected future tensor with at least 2 channels, got {tuple(future.shape)}.')
    return future[..., :2]


def _build_top_k_checkpoint_path(best_checkpoint_path: Path, epoch: int) -> Path:
    return best_checkpoint_path.with_name(
        f'{best_checkpoint_path.stem}_topk_epoch{epoch:03d}{best_checkpoint_path.suffix}'
    )


def _update_top_k_checkpoints(
    model,
    top_k_records,
    epoch_metrics,
    *,
    best_checkpoint_path: Path | None,
    top_k_checkpoints: int,
):
    if best_checkpoint_path is None or top_k_checkpoints <= 1:
        return top_k_records

    candidate_val_ade = epoch_metrics['val_ADE']
    if len(top_k_records) >= top_k_checkpoints and candidate_val_ade >= top_k_records[-1]['val_ADE']:
        return top_k_records

    checkpoint_path = _build_top_k_checkpoint_path(best_checkpoint_path, epoch_metrics['epoch'])
    torch.save(model.state_dict(), checkpoint_path)
    updated_records = [
        *top_k_records,
        {
            'epoch': epoch_metrics['epoch'],
            'val_ADE': epoch_metrics['val_ADE'],
            'val_FDE': epoch_metrics['val_FDE'],
            'val_loss': epoch_metrics['val_loss'],
            'checkpoint_path': str(checkpoint_path),
        },
    ]
    updated_records.sort(key=lambda record: (record['val_ADE'], record['epoch']))

    while len(updated_records) > top_k_checkpoints:
        removed_record = updated_records.pop(-1)
        removed_path = Path(removed_record['checkpoint_path'])
        if removed_path.exists():
            removed_path.unlink()

    return updated_records


def validate(model, val_loader, *, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_samples = 0
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0

    with torch.no_grad():
        for batch in val_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            future_xy = _get_future_xy(batch['future'].to(device))

            pred_future = model(camera, history)
            loss = F.mse_loss(pred_future, future_xy)
            ade = torch.norm(pred_future - future_xy, p=2, dim=-1).mean(dim=1).sum()
            fde = torch.norm(pred_future[:, -1, :] - future_xy[:, -1, :], p=2, dim=-1).sum()

            batch_size = future_xy.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_ade += ade.item()
            total_fde += fde.item()

    if total_samples == 0:
        raise ValueError('Validation loader produced no samples.')

    return {
        'val_loss': total_loss / total_samples,
        'val_ADE': total_ade / total_samples,
        'val_FDE': total_fde / total_samples,
    }


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    logger,
    *,
    num_epochs: int = 100,
    scheduler=None,
    scheduler_metric: str = 'val_ADE',
    best_checkpoint_path=None,
    last_checkpoint_path=None,
    top_k_checkpoints: int = 1,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    backbone_warmup_epochs: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    epoch_history = []
    best_metrics = None
    early_stopping_enabled = early_stopping_patience is not None and early_stopping_patience > 0
    backbone_warmup_enabled = backbone_warmup_epochs > 0
    best_early_stopping_ade = None
    early_stopping_bad_epochs = 0
    stopped_early = False
    best_checkpoint_path = Path(best_checkpoint_path) if best_checkpoint_path is not None else None
    last_checkpoint_path = Path(last_checkpoint_path) if last_checkpoint_path is not None else None
    top_k_checkpoint_records = []

    if best_checkpoint_path is not None:
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if last_checkpoint_path is not None:
        last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if backbone_warmup_enabled and not _supports_backbone_warmup(model, optimizer):
        raise ValueError(
            "backbone_warmup_epochs requires a model with freeze_backbone()/unfreeze_backbone() "
            "and an optimizer with a named 'backbone' parameter group."
        )

    for epoch in range(num_epochs):
        backbone_transition = _apply_backbone_warmup_state(model, epoch, backbone_warmup_epochs)
        if backbone_transition == 'frozen':
            print(f'Backbone warmup: freezing ResNet18 backbone for epoch {epoch+1}/{num_epochs}.')
        elif backbone_transition == 'unfrozen':
            print(f'Backbone warmup: unfreezing ResNet18 backbone at epoch {epoch+1}/{num_epochs}.')

        model.train()
        total_samples = 0
        train_loss_sum = 0.0

        for idx, batch in enumerate(train_loader):
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            future_xy = _get_future_xy(batch['future'].to(device))

            optimizer.zero_grad()
            pred_future = model(camera, history)
            loss = F.mse_loss(pred_future, future_xy)
            loss.backward()
            optimizer.step()

            batch_size = future_xy.size(0)
            total_samples += batch_size
            train_loss_sum += loss.item() * batch_size

            if idx % 10 == 0:
                logger.log(step=epoch * len(train_loader) + idx, loss=loss.item())

        if total_samples == 0:
            raise ValueError('Training loader produced no samples.')

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss_sum / total_samples,
            'backbone_frozen': bool(getattr(model, 'backbone_is_frozen', False)),
        }
        epoch_metrics.update(validate(model, val_loader, device=device))

        if best_metrics is None or epoch_metrics['val_ADE'] < best_metrics['val_ADE']:
            best_metrics = dict(epoch_metrics)
            if best_checkpoint_path is not None:
                torch.save(model.state_dict(), best_checkpoint_path)
                best_metrics['checkpoint_path'] = str(best_checkpoint_path)

        top_k_checkpoint_records = _update_top_k_checkpoints(
            model,
            top_k_checkpoint_records,
            epoch_metrics,
            best_checkpoint_path=best_checkpoint_path,
            top_k_checkpoints=top_k_checkpoints,
        )

        if last_checkpoint_path is not None:
            torch.save(model.state_dict(), last_checkpoint_path)
            epoch_metrics['last_checkpoint_path'] = str(last_checkpoint_path)

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
        last_checkpoint_path=str(last_checkpoint_path) if last_checkpoint_path is not None else None,
        top_k_checkpoint_paths=[record['checkpoint_path'] for record in top_k_checkpoint_records],
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
        'best_checkpoint_path': str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        'last_checkpoint_path': str(last_checkpoint_path) if last_checkpoint_path is not None else None,
        'top_k_checkpoints': top_k_checkpoints,
        'top_k_checkpoint_paths': [record['checkpoint_path'] for record in top_k_checkpoint_records],
        'top_k_checkpoint_records': top_k_checkpoint_records,
    }
