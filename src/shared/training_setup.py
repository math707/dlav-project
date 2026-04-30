import torch.optim as optim


def build_optimizer(
    model,
    learning_rate: float,
    weight_decay: float = 0.0,
    backbone_learning_rate: float | None = None,
    backbone_lr_scale: float | None = None,
):
    use_backbone_specific_lr = backbone_learning_rate is not None or backbone_lr_scale is not None

    if hasattr(model, 'get_optimizer_param_groups'):
        parameter_groups = model.get_optimizer_param_groups(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            backbone_learning_rate=backbone_learning_rate,
            backbone_lr_scale=backbone_lr_scale,
        )
        return optim.Adam(parameter_groups, lr=learning_rate, weight_decay=weight_decay)

    if use_backbone_specific_lr:
        raise ValueError(
            "This model does not expose separate backbone parameter groups. "
            "Use the backbone-specific learning-rate options with model_b or model_b_v2."
        )

    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_scheduler(
    optimizer,
    enabled: bool,
    name: str = 'plateau',
    factor: float = 0.5,
    patience: int = 4,
    min_lr: float = 1e-5,
):
    if not enabled:
        return None

    if name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(f"Unsupported scheduler '{name}'.")
