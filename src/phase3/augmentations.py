"""Safe photometric augmentations for Phase 3 sim-to-real training."""

from __future__ import annotations


def _require_torchvision_transforms():
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            'Phase 3 image augmentations require torchvision. Install torchvision or disable augmentation.'
        ) from exc
    return transforms


def _clamp_strength(strength: float) -> float:
    strength = float(strength)
    if strength < 0.0:
        return 0.0
    if strength > 1.0:
        return 1.0
    return strength


def build_train_augmentations(strength: float = 0.5):
    """Build sim-to-real augmentations without geometry changes."""

    transforms = _require_torchvision_transforms()
    strength = _clamp_strength(strength)

    brightness = 0.05 + 0.25 * strength
    contrast = 0.05 + 0.25 * strength
    saturation = 0.05 + 0.20 * strength
    hue = 0.01 + 0.03 * strength
    blur_sigma_max = 0.25 + 1.0 * strength

    return transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                    )
                ],
                p=0.55,
            ),
            transforms.RandomGrayscale(p=0.05 + 0.10 * strength),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, blur_sigma_max))],
                p=0.10 + 0.15 * strength,
            ),
            transforms.RandomAutocontrast(p=0.05 + 0.15 * strength),
        ]
    )
