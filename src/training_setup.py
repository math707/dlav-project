import torch.optim as optim


def build_optimizer(model, learning_rate: float, weight_decay: float = 0.0):
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
