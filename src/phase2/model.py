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
            "MODEL_NAME='phase2_trajectory_only', 'phase2_multitask', 'phase2_b_v2_depth', or "
            "'phase2_gru_delta_residual', or 'phase2_temporal_delta_depth'."
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


def _build_resnet18_pooled_backbone(pretrained: bool, strict_pretrained: bool = False):
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise ImportError(
            "Phase 2 ResNet18 planner variants require torchvision. Install torchvision to use "
            "MODEL_NAME='phase2_trajectory_only', 'phase2_multitask', 'phase2_b_v2_port', or "
            "'phase2_b_v2_depth'."
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

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
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


class TemporalConvResidualBlock(nn.Module):
    """Residual temporal Conv1D block over future-step tokens."""

    def __init__(self, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return residual + x


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


class Phase2ModelBV2Port(nn.Module):
    """Phase 1 model_b_v2-style Phase 2 ablation without command conditioning."""

    default_backbone_lr_scale = 0.1

    def __init__(
        self,
        *,
        history_steps: int = 21,
        history_features: int = 3,
        future_steps: int = 60,
        trajectory_features: int = 2,
        visual_embedding_dim: int = 256,
        history_hidden_dim: int = 128,
        history_embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        normalize_camera: bool = True,
        strict_pretrained_backbone: bool = True,
        keep_backbone_in_eval_when_frozen: bool = True,
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
        self.visual_backbone, visual_feature_dim = _build_resnet18_pooled_backbone(
            pretrained=pretrained_backbone,
            strict_pretrained=strict_pretrained_backbone,
        )
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

        self.fusion_head = nn.Sequential(
            nn.Linear(visual_embedding_dim + history_embedding_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trajectory_head = nn.Linear(fusion_hidden_dim, future_steps * trajectory_features)

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def backbone_is_frozen(self):
        return not self._backbone_trainable

    @property
    def supports_depth_aux(self) -> bool:
        return False

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

    def get_optimizer_param_groups(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        backbone_learning_rate: float | None = None,
        backbone_lr_scale: float | None = None,
    ):
        if backbone_learning_rate is not None and backbone_lr_scale is not None:
            raise ValueError("Specify either backbone_learning_rate or backbone_lr_scale, not both.")

        head_modules = [self.visual_projection, self.history_encoder, self.fusion_head, self.trajectory_head]
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

    def forward(self, camera, history, driving_command=None, return_aux: bool = False):
        del driving_command

        visual_embedding = self.visual_projection(self.visual_backbone(self._prepare_camera(camera)))

        history_flat = history.reshape(history.size(0), -1)
        history_embedding = self.history_encoder(history_flat)

        fused_features = torch.cat([visual_embedding, history_embedding], dim=1)
        trajectory = self.trajectory_head(self.fusion_head(fused_features))
        trajectory = trajectory.reshape(-1, self.future_steps, self.trajectory_features)

        if return_aux:
            return {'trajectory': trajectory}
        return trajectory


class Phase2ModelBV2Depth(nn.Module):
    """Phase 1 model_b_v2-style Phase 2 ablation with depth supervision and no command conditioning."""

    default_backbone_lr_scale = 0.1

    def __init__(
        self,
        *,
        history_steps: int = 21,
        history_features: int = 3,
        future_steps: int = 60,
        trajectory_features: int = 2,
        visual_embedding_dim: int = 256,
        history_hidden_dim: int = 128,
        history_embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        normalize_camera: bool = True,
        strict_pretrained_backbone: bool = True,
        keep_backbone_in_eval_when_frozen: bool = True,
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

        self.fusion_head = nn.Sequential(
            nn.Linear(visual_embedding_dim + history_embedding_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trajectory_head = nn.Linear(fusion_hidden_dim, future_steps * trajectory_features)
        self.depth_head = DepthDecoder(visual_feature_dim)

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def backbone_is_frozen(self):
        return not self._backbone_trainable

    @property
    def supports_depth_aux(self) -> bool:
        return True

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

        head_modules = [self.visual_projection, self.history_encoder, self.fusion_head, self.trajectory_head, self.depth_head]
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

    def forward(self, camera, history, driving_command=None, return_aux: bool = False):
        del driving_command

        feature_map, visual_embedding = self._encode_visual(camera)

        history_flat = history.reshape(history.size(0), -1)
        history_embedding = self.history_encoder(history_flat)

        fused_features = torch.cat([visual_embedding, history_embedding], dim=1)
        trajectory = self.trajectory_head(self.fusion_head(fused_features))
        trajectory = trajectory.reshape(-1, self.future_steps, self.trajectory_features)

        if not return_aux:
            return trajectory

        return {
            'trajectory': trajectory,
            'depth': self.depth_head(feature_map, output_size=camera.shape[-2:]),
        }


class Phase2GRUDeltaResidualPlanner(nn.Module):
    """Sequence-aware planner with GRU history encoding and GRU delta-residual decoding.

    The decoder predicts a residual `delta_xy` over a constant-velocity prior computed from the
    last two history XY points. The final trajectory is obtained by cumulatively summing the
    resulting deltas from the last observed XY position.
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
        decoder_hidden_dim: int = 256,
        decoder_context_dim: int = 64,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        normalize_camera: bool = True,
        strict_pretrained_backbone: bool = True,
        keep_backbone_in_eval_when_frozen: bool = True,
        use_depth_head: bool = True,
    ):
        super().__init__()
        if trajectory_features != 2:
            raise ValueError('phase2_gru_delta_residual expects trajectory_features=2 for XY decoding.')
        if history_steps < 2:
            raise ValueError('phase2_gru_delta_residual requires at least two history steps.')

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

        self.history_encoder = nn.GRU(
            input_size=history_features,
            hidden_size=history_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.history_projection = nn.Sequential(
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

        self.decoder_init = nn.Sequential(
            nn.Linear(fusion_hidden_dim, decoder_hidden_dim),
            nn.Tanh(),
        )
        self.decoder_context = nn.Sequential(
            nn.Linear(fusion_hidden_dim, decoder_context_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.GRU(
            input_size=trajectory_features + decoder_context_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.residual_head = nn.Linear(decoder_hidden_dim, trajectory_features)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

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

    def _encode_history(self, history):
        _, history_hidden = self.history_encoder(history)
        history_hidden = history_hidden[-1]
        return self.history_projection(history_hidden)

    def _encode_command(self, driving_command, batch_size: int, device):
        if driving_command is None:
            return torch.zeros(batch_size, self.command_encoder[0].out_features, device=device)

        driving_command = torch.as_tensor(driving_command, device=device).long().view(-1)
        if driving_command.numel() == 1 and batch_size > 1:
            driving_command = driving_command.expand(batch_size)
        if driving_command.numel() != batch_size:
            raise ValueError(
                f'driving_command batch has {driving_command.numel()} entries for batch size {batch_size}.'
            )
        return self.command_encoder(self.command_embedding(driving_command))

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
            self.history_projection,
            self.command_embedding,
            self.command_encoder,
            self.fusion_head,
            self.decoder_init,
            self.decoder_context,
            self.decoder,
            self.residual_head,
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

    def forward(self, camera, history, driving_command=None, return_aux: bool = False):
        feature_map, visual_embedding = self._encode_visual(camera)

        history = history.float()
        history_embedding = self._encode_history(history)
        command_embedding = self._encode_command(driving_command, batch_size=history.size(0), device=history.device)

        fused_features = torch.cat([visual_embedding, history_embedding, command_embedding], dim=1)
        fused_context = self.fusion_head(fused_features)

        decoder_hidden = self.decoder_init(fused_context).unsqueeze(0)
        decoder_context = self.decoder_context(fused_context)

        history_xy = history[..., :2]
        last_history_xy = history_xy[:, -1, :]
        prior_velocity = history_xy[:, -1, :] - history_xy[:, -2, :]

        prev_delta = prior_velocity
        predicted_deltas = []
        for _ in range(self.future_steps):
            decoder_input = torch.cat([prev_delta, decoder_context], dim=1).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            residual_delta = self.residual_head(decoder_output.squeeze(1))
            step_delta = prior_velocity + residual_delta
            predicted_deltas.append(step_delta)
            prev_delta = step_delta

        delta_sequence = torch.stack(predicted_deltas, dim=1)
        trajectory = last_history_xy.unsqueeze(1) + torch.cumsum(delta_sequence, dim=1)

        if not return_aux:
            return trajectory

        outputs = {'trajectory': trajectory}
        if self.depth_head is not None:
            outputs['depth'] = self.depth_head(feature_map, output_size=camera.shape[-2:])
        return outputs


class Phase2TemporalDeltaDepthPlanner(nn.Module):
    """Parallel timestep-conditioned delta-residual decoder on top of the best Phase 2 encoder family."""

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
        timestep_embedding_dim: int = 64,
        temporal_hidden_dim: int = 256,
        temporal_kernel_size: int = 5,
        temporal_blocks: int = 3,
        temporal_mlp_hidden_dim: int = 128,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        normalize_camera: bool = True,
        strict_pretrained_backbone: bool = True,
        keep_backbone_in_eval_when_frozen: bool = True,
        use_depth_head: bool = True,
    ):
        super().__init__()
        if trajectory_features != 2:
            raise ValueError('phase2_temporal_delta_depth expects trajectory_features=2 for XY decoding.')
        if history_steps < 2:
            raise ValueError('phase2_temporal_delta_depth requires at least two history steps.')

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

        self.timestep_embedding = nn.Embedding(future_steps, timestep_embedding_dim)
        self.register_buffer('future_step_indices', torch.arange(future_steps, dtype=torch.long), persistent=False)
        self.register_buffer(
            'normalized_future_steps',
            torch.linspace(0.0, 1.0, steps=future_steps, dtype=torch.float32).view(1, future_steps, 1),
            persistent=False,
        )

        temporal_input_dim = fusion_hidden_dim + timestep_embedding_dim + trajectory_features + 1
        self.temporal_input_projection = nn.Sequential(
            nn.Linear(temporal_input_dim, temporal_hidden_dim),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList(
            [TemporalConvResidualBlock(temporal_hidden_dim, kernel_size=temporal_kernel_size) for _ in range(temporal_blocks)]
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(temporal_hidden_dim),
            nn.Linear(temporal_hidden_dim, temporal_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(temporal_mlp_hidden_dim, trajectory_features),
        )

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

    def _encode_command(self, driving_command, batch_size: int, device):
        if driving_command is None:
            return torch.zeros(batch_size, self.command_encoder[0].out_features, device=device)

        driving_command = torch.as_tensor(driving_command, device=device).long().view(-1)
        if driving_command.numel() == 1 and batch_size > 1:
            driving_command = driving_command.expand(batch_size)
        if driving_command.numel() != batch_size:
            raise ValueError(
                f'driving_command batch has {driving_command.numel()} entries for batch size {batch_size}.'
            )
        return self.command_encoder(self.command_embedding(driving_command))

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
            self.timestep_embedding,
            self.temporal_input_projection,
            self.temporal_blocks,
            self.delta_head,
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

    def forward(self, camera, history, driving_command=None, return_aux: bool = False):
        feature_map, visual_embedding = self._encode_visual(camera)

        history = history.float()
        history_flat = history.reshape(history.size(0), -1)
        history_embedding = self.history_encoder(history_flat)
        command_embedding = self._encode_command(driving_command, batch_size=history.size(0), device=history.device)

        fused_features = torch.cat([visual_embedding, history_embedding, command_embedding], dim=1)
        global_context = self.fusion_head(fused_features)

        history_xy = history[..., :2]
        last_history_xy = history_xy[:, -1, :]
        prior_velocity = history_xy[:, -1, :] - history_xy[:, -2, :]

        repeated_context = global_context.unsqueeze(1).expand(-1, self.future_steps, -1)
        timestep_embedding = self.timestep_embedding(self.future_step_indices).unsqueeze(0).expand(history.size(0), -1, -1)
        repeated_prior_velocity = prior_velocity.unsqueeze(1).expand(-1, self.future_steps, -1)
        repeated_future_steps = self.normalized_future_steps.expand(history.size(0), -1, -1)

        temporal_tokens = torch.cat(
            [repeated_context, timestep_embedding, repeated_prior_velocity, repeated_future_steps],
            dim=-1,
        )
        temporal_features = self.temporal_input_projection(temporal_tokens)
        for block in self.temporal_blocks:
            temporal_features = block(temporal_features)

        residual_delta = self.delta_head(temporal_features)
        predicted_delta = repeated_prior_velocity + residual_delta
        trajectory = last_history_xy.unsqueeze(1) + torch.cumsum(predicted_delta, dim=1)

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
    'phase2_b_v2_port': Phase2ModelBV2Port,
    'phase2_b_v2_depth': Phase2ModelBV2Depth,
    'phase2_gru_delta_residual': Phase2GRUDeltaResidualPlanner,
    'phase2_temporal_delta_depth': Phase2TemporalDeltaDepthPlanner,
}


def build_model(name: str = 'phase2_multitask', **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_cls(**kwargs)
