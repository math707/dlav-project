"""Phase 2 training helpers with optional depth auxiliary supervision."""

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


def _unpack_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs.get('trajectory'), outputs.get('depth')
    if isinstance(outputs, tuple):
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        if len(outputs) == 1:
            return outputs[0], None
    return outputs, None


def _get_future_xy(future):
    return future[..., :2]


def _compute_depth_loss(pred_depth, target_depth, depth_loss_name: str):
    if depth_loss_name == 'l1':
        return F.l1_loss(pred_depth, target_depth)
    if depth_loss_name == 'smooth_l1':
        return F.smooth_l1_loss(pred_depth, target_depth)
    raise ValueError(f"Unsupported depth loss '{depth_loss_name}'.")


def validate(
    model,
    val_loader,
    *,
    device=None,
    use_depth_aux: bool = False,
    lambda_depth: float = 0.05,
    depth_loss_name: str = 'l1',
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_samples = 0
    total_traj_loss = 0.0
    total_depth_loss = 0.0
    total_total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0

    with torch.no_grad():
        for batch in val_loader:
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            driving_command = batch['driving_command'].to(device)
            future_xy = _get_future_xy(batch['future'].to(device))

            outputs = model(camera, history, driving_command, return_aux=use_depth_aux)
            pred_future, pred_depth = _unpack_model_outputs(outputs)
            if pred_future is None:
                raise RuntimeError('Model did not return a trajectory prediction.')

            batch_size = future_xy.size(0)
            traj_loss = F.mse_loss(pred_future, future_xy)
            total_loss = traj_loss
            depth_loss = None

            if use_depth_aux:
                if 'depth' not in batch:
                    raise KeyError('Depth supervision is enabled, but the validation batch has no depth tensor.')
                if pred_depth is None:
                    raise RuntimeError('Depth supervision is enabled, but the model did not return a depth prediction.')
                target_depth = batch['depth'].to(device)
                depth_loss = _compute_depth_loss(pred_depth, target_depth, depth_loss_name)
                total_loss = total_loss + lambda_depth * depth_loss

            ade = torch.norm(pred_future - future_xy, p=2, dim=-1).mean(dim=1).sum()
            fde = torch.norm(pred_future[:, -1, :] - future_xy[:, -1, :], p=2, dim=-1).sum()

            total_samples += batch_size
            total_traj_loss += traj_loss.item() * batch_size
            total_total_loss += total_loss.item() * batch_size
            total_ade += ade.item()
            total_fde += fde.item()
            if depth_loss is not None:
                total_depth_loss += depth_loss.item() * batch_size

    if total_samples == 0:
        raise ValueError('Validation loader produced no samples.')

    depth_loss_value = total_depth_loss / total_samples if use_depth_aux else None
    return {
        'val_traj_loss': total_traj_loss / total_samples,
        'val_depth_loss': depth_loss_value,
        'val_total_loss': total_total_loss / total_samples,
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
    num_epochs: int = 50,
    scheduler=None,
    scheduler_metric: str = 'val_ADE',
    best_checkpoint_path=None,
    lambda_depth: float = 0.05,
    use_depth_aux: bool = False,
    depth_loss_name: str = 'l1',
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

        model.train()
        total_samples = 0
        train_traj_loss_sum = 0.0
        train_depth_loss_sum = 0.0
        train_total_loss_sum = 0.0

        for idx, batch in enumerate(train_loader):
            camera = batch['camera'].to(device)
            history = batch['history'].to(device)
            driving_command = batch['driving_command'].to(device)
            future_xy = _get_future_xy(batch['future'].to(device))

            optimizer.zero_grad()
            outputs = model(camera, history, driving_command, return_aux=use_depth_aux)
            pred_future, pred_depth = _unpack_model_outputs(outputs)
            if pred_future is None:
                raise RuntimeError('Model did not return a trajectory prediction.')

            traj_loss = F.mse_loss(pred_future, future_xy)
            total_loss = traj_loss
            depth_loss = None

            if use_depth_aux:
                if 'depth' not in batch:
                    raise KeyError('Depth supervision is enabled, but the training batch has no depth tensor.')
                if pred_depth is None:
                    raise RuntimeError('Depth supervision is enabled, but the model did not return a depth prediction.')
                target_depth = batch['depth'].to(device)
                depth_loss = _compute_depth_loss(pred_depth, target_depth, depth_loss_name)
                total_loss = total_loss + lambda_depth * depth_loss

            total_loss.backward()
            optimizer.step()

            batch_size = future_xy.size(0)
            total_samples += batch_size
            train_traj_loss_sum += traj_loss.item() * batch_size
            train_total_loss_sum += total_loss.item() * batch_size
            if depth_loss is not None:
                train_depth_loss_sum += depth_loss.item() * batch_size

            if idx % 10 == 0:
                batch_metrics = {
                    'train_total_loss': total_loss.item(),
                    'train_traj_loss': traj_loss.item(),
                }
                if depth_loss is not None:
                    batch_metrics['train_depth_loss'] = depth_loss.item()
                logger.log(step=epoch * len(train_loader) + idx, **batch_metrics)

        if total_samples == 0:
            raise ValueError('Training loader produced no samples.')

        train_depth_loss_value = train_depth_loss_sum / total_samples if use_depth_aux else None
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_total_loss': train_total_loss_sum / total_samples,
            'train_traj_loss': train_traj_loss_sum / total_samples,
            'train_depth_loss': train_depth_loss_value,
            'backbone_frozen': bool(getattr(model, 'backbone_is_frozen', False)),
        }
        epoch_metrics.update(
            validate(
                model,
                val_loader,
                device=device,
                use_depth_aux=use_depth_aux,
                lambda_depth=lambda_depth,
                depth_loss_name=depth_loss_name,
            )
        )

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

        depth_display = ''
        if use_depth_aux and epoch_metrics['train_depth_loss'] is not None and epoch_metrics['val_depth_loss'] is not None:
            depth_display = (
                f" | Train Depth: {epoch_metrics['train_depth_loss']:.4f}"
                f" | Val Depth: {epoch_metrics['val_depth_loss']:.4f}"
            )
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Total: {epoch_metrics['train_total_loss']:.4f} | "
            f"Train Traj: {epoch_metrics['train_traj_loss']:.4f} | "
            f"Val Total: {epoch_metrics['val_total_loss']:.4f} | "
            f"Val Traj: {epoch_metrics['val_traj_loss']:.4f} | "
            f"ADE: {epoch_metrics['val_ADE']:.4f} | "
            f"FDE: {epoch_metrics['val_FDE']:.4f}"
            f"{depth_display} | "
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
        use_depth_aux=use_depth_aux,
        lambda_depth=lambda_depth,
        depth_loss_name=depth_loss_name,
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
        'use_depth_aux': use_depth_aux,
        'lambda_depth': lambda_depth,
        'depth_loss_name': depth_loss_name,
    }
