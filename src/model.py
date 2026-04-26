import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn


class DrivingPlanner(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN for processing camera images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Decoder for predicting future trajectory
        self.decoder = nn.Linear(200 * 300 // 4 * 32 + 21 * 3, 60 * 3)

    def forward(self, camera, history):
        # Process camera images
        visual_features = self.cnn(camera)

        # Combine features
        history_flat = history.reshape(history.size(0), -1)
        combined = torch.cat([visual_features, history_flat], dim=1)

        # Predict future trajectory
        future = self.decoder(combined)
        future = future.reshape(-1, 60, 3)  # Reshape to (batch_size, timesteps, features)

        return future


DrivingPlannerBaseline = DrivingPlanner


def _default_torch_home() -> Path:
    project_root = Path(__file__).resolve().parents[1]
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


def _build_resnet18_backbone(pretrained: bool, strict_pretrained: bool = False):
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise ImportError(
            "ResNet18 planner variants require torchvision. Install torchvision to use MODEL_NAME='model_b' or 'model_b_v2'."
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

    feature_dim = backbone.fc.in_features
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
    return feature_extractor, feature_dim


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


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DrivingPlannerModelA(nn.Module):
    def __init__(
        self,
        history_steps=21,
        history_features=3,
        future_steps=60,
        future_features=3,
        visual_embedding_dim=256,
        history_hidden_dim=128,
        history_embedding_dim=128,
        fusion_hidden_dim=256,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.future_features = future_features

        # Keep spatial structure through several stages, then pool to a compact embedding.
        self.visual_encoder = nn.Sequential(
            ConvEncoderBlock(3, 32, stride=2),
            ConvEncoderBlock(32, 64, stride=2),
            ConvEncoderBlock(64, 128, stride=2),
            nn.Conv2d(128, visual_embedding_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(visual_embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        history_input_dim = history_steps * history_features
        self.history_encoder = nn.Sequential(
            nn.Linear(history_input_dim, history_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(history_hidden_dim, history_embedding_dim),
            nn.ReLU(inplace=True),
        )

        # A small fusion MLP is easier to optimize than one giant decoder layer.
        self.fusion_head = nn.Sequential(
            nn.Linear(visual_embedding_dim + history_embedding_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Linear(fusion_hidden_dim, future_steps * future_features)

    def forward(self, camera, history):
        visual_embedding = self.visual_encoder(camera)

        history_flat = history.reshape(history.size(0), -1)
        history_embedding = self.history_encoder(history_flat)

        fused_features = torch.cat([visual_embedding, history_embedding], dim=1)
        future = self.output_layer(self.fusion_head(fused_features))

        return future.reshape(-1, self.future_steps, self.future_features)


class DrivingPlannerResNet18Planner(nn.Module):
    default_backbone_lr_scale = None

    def __init__(
        self,
        history_steps=21,
        history_features=3,
        future_steps=60,
        future_features=3,
        visual_embedding_dim=256,
        history_hidden_dim=128,
        history_embedding_dim=128,
        fusion_hidden_dim=256,
        pretrained_backbone=True,
        freeze_backbone=False,
        normalize_camera=True,
        strict_pretrained_backbone=False,
        keep_backbone_in_eval_when_frozen=True,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.future_features = future_features
        self.keep_backbone_in_eval_when_frozen = keep_backbone_in_eval_when_frozen
        self._backbone_trainable = True

        self.camera_preprocessor = CameraTensorPreprocessor(
            normalize_to_unit_scale=True,
            imagenet_normalize=normalize_camera,
        )
        self.visual_backbone, visual_feature_dim = _build_resnet18_backbone(
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
        self.output_layer = nn.Linear(fusion_hidden_dim, future_steps * future_features)

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def backbone_is_frozen(self):
        return not self._backbone_trainable

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

        head_modules = [self.visual_projection, self.history_encoder, self.fusion_head, self.output_layer]
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
                    "name": "head",
                    "params": head_parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
        if backbone_parameters:
            parameter_groups.append(
                {
                    "name": "backbone",
                    "params": backbone_parameters,
                    "lr": resolved_backbone_lr,
                    "weight_decay": weight_decay,
                }
            )

        return parameter_groups

    def forward(self, camera, history):
        visual_embedding = self.visual_projection(self.visual_backbone(self._prepare_camera(camera)))

        history_flat = history.reshape(history.size(0), -1)
        history_embedding = self.history_encoder(history_flat)

        fused_features = torch.cat([visual_embedding, history_embedding], dim=1)
        future = self.output_layer(self.fusion_head(fused_features))

        return future.reshape(-1, self.future_steps, self.future_features)


class DrivingPlannerModelB(DrivingPlannerResNet18Planner):
    pass


class DrivingPlannerModelBV2(DrivingPlannerResNet18Planner):
    default_backbone_lr_scale = 0.1

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('strict_pretrained_backbone', True)
        super().__init__(*args, **kwargs)


MODEL_REGISTRY = {
    'baseline': DrivingPlanner,
    'model_a': DrivingPlannerModelA,
    'model_b': DrivingPlannerModelB,
    'model_b_v2': DrivingPlannerModelBV2,
}


def build_model(name='baseline', **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_cls(**kwargs)
