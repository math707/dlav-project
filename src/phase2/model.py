"""Phase 2 models for perception-aware planning."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import NUM_DRIVING_COMMANDS


def _default_torch_home() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidate_dirs: list[Path] = []

    if os.name == 'nt':
        base_cache_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        candidate_dirs.append(base_cache_dir / 'dlav-project' / 'torch_cache')
    else:
        xdg_cache_home = os.environ.get('XDG_CACHE_HOME')
        base_cache_dir = Path(xdg_cache_home) if xdg_cache_home else Path.home() / '.cache'
        candidate_dirs.append(base_cache_dir / 'dlav-project' / 'torch_cache')

    candidate_dirs.append(project_root / 'outputs' / 'torch_cache')

    for cache_dir in candidate_dirs:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        except OSError:
            continue

    raise RuntimeError('Failed to find a writable torch cache directory for pretrained weights.')


def _ensure_torch_home():
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = str(_default_torch_home())


def _collect_parameters(modules, trainable_only: bool = True):
    parameters = []
    seen_parameter_ids = set()

    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if (not trainable_only or parameter.requires_grad) and parameter_id not in seen_parameter_ids:
                parameters.append(parameter)
                seen_parameter_ids.add(parameter_id)

    return parameters


def _build_resnet18_spatial_backbone(pretrained: bool, strict_pretrained: bool = False):
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise ImportError(
            "Phase 2 ResNet18 planner variants require torchvision. Install torchvision to use "
            "MODEL_NAME='phase2_trajectory_only' or 'phase2_multitask'."
        ) from exc

    backbone = None
    if pretrained:
        _ensure_torch_home()
        try:
            from torchvision.models import ResNet18_Weights

            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            backbone = resnet18(pretrained=True)
        except Exception as exc:
            if strict_pretrained:
                torch_home = os.environ.get('TORCH_HOME', str(_default_torch_home()))
                raise RuntimeError(
                    "Failed to load pretrained ResNet18 weights. "
                    f"Ensure a writable torch cache is available (TORCH_HOME={torch_home}) "
                    f"or set pretrained_backbone=False. Original error: {exc}"
                ) from exc
            warnings.warn(
                f"Failed to load pretrained ResNet18 weights ({exc}). Falling back to randomly initialized weights.",
                RuntimeWarning,
            )

    if backbone is None:
        try:
            backbone = resnet18(weights=None)
        except TypeError:
            backbone = resnet18(pretrained=False)

    feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
    return feature_extractor, backbone.fc.in_features


class CameraTensorPreprocessor(nn.Module):
    def __init__(self, normalize_to_unit_scale: bool = True, imagenet_normalize: bool = False):
        super().__init__()
        self.normalize_to_unit_scale = normalize_to_unit_scale
        self.imagenet_normalize = imagenet_normalize

        self.register_buffer(
            'camera_mean',
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'camera_std',
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, camera):
        camera = camera.float()

        if self.normalize_to_unit_scale and camera.detach().amax().item() > 1.5:
            camera = camera / 255.0

        if self.imagenet_normalize:
            camera = (camera - self.camera_mean) / self.camera_std

        return camera


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthDecoder(nn.Module):
    """Lightweight decoder that upsamples ResNet18 feature maps into depth maps."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.block1 = UpsampleConvBlock(in_channels, 256)
        self.block2 = UpsampleConvBlock(256, 128)
        self.block3 = UpsampleConvBlock(128, 64)
        self.block4 = UpsampleConvBlock(64, 32)
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feature_map, output_size: tuple[int, int]):
        x = F.interpolate(feature_map, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block4(x)
        x = self.output_layer(x)
        return F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)


class Phase2ResNet18Planner(nn.Module):
    """ResNet18-based Phase 2 planner with command conditioning and optional depth supervision.

    Inputs:
    - `camera`: `[batch, 3, height, width]`
    - `history`: `[batch, 21, 3]`
    - `driving_command`: `[batch]` integer ids in `{0, 1, 2}`

    Outputs:
    - default: trajectory tensor `[batch, 60, 2]`
    - with `return_aux=True` and a depth head: `{'trajectory': [batch, 60, 2], 'depth': [batch, 1, height, width]}`
    """

    default_backbone_lr_scale = 0.1

    def __init__(
        self,
        *,
        history_steps: int = 21,
        history_features: int = 3,
        future_steps: int = 60,
        trajectory_features: int = 2,
        command_vocab_size: int = NUM_DRIVING_COMMANDS,
        command_embedding_dim: int = 16,
        command_hidden_dim: int = 32,
        visual_embedding_dim: int = 256,
        history_hidden_dim: int = 128,
        history_embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        normalize_camera: bool = True,
        strict_pretrained_backbone: bool = True,
        keep_backbone_in_eval_when_frozen: bool = True,
        use_depth_head: bool = False,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.trajectory_features = trajectory_features
        self.keep_backbone_in_eval_when_frozen = keep_backbone_in_eval_when_frozen
        self._backbone_trainable = True

        self.camera_preprocessor = CameraTensorPreprocessor(
            normalize_to_unit_scale=True,
            imagenet_normalize=normalize_camera,
        )
        self.visual_backbone, visual_feature_dim = _build_resnet18_spatial_backbone(
            pretrained=pretrained_backbone,
            strict_pretrained=strict_pretrained_backbone,
        )
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_feature_dim, visual_embedding_dim),
            nn.ReLU(inplace=True),
        )

        history_input_dim = history_steps * history_features
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, history_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(history_hidden_dim, history_embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.command_embedding = nn.Embedding(command_vocab_size, command_embedding_dim)
        self.command_encoder = nn.Sequential(
            nn.Linear(command_embedding_dim, command_hidden_dim),
            nn.ReLU(inplace=True),
        )

        fused_dim = visual_embedding_dim + history_embedding_dim + command_hidden_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trajectory_head = nn.Linear(fusion_hidden_dim, future_steps * trajectory_features)
        self.depth_head = DepthDecoder(visual_feature_dim) if use_depth_head else None

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def backbone_is_frozen(self):
        return not self._backbone_trainable

    @property
    def supports_depth_aux(self) -> bool:
        return self.depth_head is not None

    def set_backbone_trainable(self, trainable: bool):
        self._backbone_trainable = trainable
        for parameter in self.visual_backbone.parameters():
            parameter.requires_grad = trainable
        return self

    def freeze_backbone(self):
        return self.set_backbone_trainable(False)

    def unfreeze_backbone(self):
        return self.set_backbone_trainable(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.backbone_is_frozen and self.keep_backbone_in_eval_when_frozen:
            self.visual_backbone.eval()
        return self

    def _prepare_camera(self, camera):
        return self.camera_preprocessor(camera)

    def _encode_visual(self, camera):
        feature_map = self.visual_backbone(self._prepare_camera(camera))
        pooled = self.visual_pool(feature_map).flatten(1)
        visual_embedding = self.visual_projection(pooled)
        return feature_map, visual_embedding

    def get_optimizer_param_groups(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        backbone_learning_rate: float | None = None,
        backbone_lr_scale: float | None = None,
    ):
        if backbone_learning_rate is not None and backbone_lr_scale is not None:
            raise ValueError("Specify either backbone_learning_rate or backbone_lr_scale, not both.")

        head_modules = [
            self.visual_projection,
            self.history_encoder,
            self.command_embedding,
            self.command_encoder,
            self.fusion_head,
            self.trajectory_head,
            self.depth_head,
        ]
        backbone_parameters = _collect_parameters([self.visual_backbone], trainable_only=False)
        head_parameters = _collect_parameters(head_modules, trainable_only=True)

        if backbone_learning_rate is None and backbone_lr_scale is None:
            backbone_lr_scale = self.default_backbone_lr_scale
        resolved_backbone_lr = (
            backbone_learning_rate
            if backbone_learning_rate is not None
            else learning_rate if backbone_lr_scale is None else learning_rate * backbone_lr_scale
        )

        parameter_groups = []
        if head_parameters:
            parameter_groups.append(
                {
                    'name': 'head',
                    'params': head_parameters,
                    'lr': learning_rate,
                    'weight_decay': weight_decay,
                }
            )
        if backbone_parameters:
            parameter_groups.append(
                {
                    'name': 'backbone',
                    'params': backbone_parameters,
                    'lr': resolved_backbone_lr,
                    'weight_decay': weight_decay,
                }
            )

        return parameter_groups

    def forward(self, camera, history, driving_command, return_aux: bool = False):
        feature_map, visual_embedding = self._encode_visual(camera)

        history_flat = history.reshape(history.size(0), -1)
        history_embedding = self.history_encoder(history_flat)

        driving_command = driving_command.long().view(-1)
        command_embedding = self.command_encoder(self.command_embedding(driving_command))

        fused_features = torch.cat([visual_embedding, history_embedding, command_embedding], dim=1)
        trajectory = self.trajectory_head(self.fusion_head(fused_features))
        trajectory = trajectory.reshape(-1, self.future_steps, self.trajectory_features)

        if not return_aux:
            return trajectory

        outputs = {'trajectory': trajectory}
        if self.depth_head is not None:
            outputs['depth'] = self.depth_head(feature_map, output_size=camera.shape[-2:])
        return outputs


class Phase2TrajectoryOnlyPlanner(Phase2ResNet18Planner):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('use_depth_head', False)
        super().__init__(*args, **kwargs)


class Phase2MultiTaskPlanner(Phase2ResNet18Planner):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('use_depth_head', True)
        super().__init__(*args, **kwargs)


MODEL_REGISTRY = {
    'phase2_trajectory_only': Phase2TrajectoryOnlyPlanner,
    'phase2_multitask': Phase2MultiTaskPlanner,
}


def build_model(name: str = 'phase2_multitask', **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_cls(**kwargs)
