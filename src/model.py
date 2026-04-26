import warnings

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


def _build_resnet18_backbone(pretrained: bool):
    try:
        from torchvision.models import resnet18
    except ImportError as exc:
        raise ImportError(
            "DrivingPlannerModelB requires torchvision. Install torchvision to use MODEL_NAME='model_b'."
        ) from exc

    backbone = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            backbone = resnet18(pretrained=True)
        except Exception as exc:
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


def _collect_trainable_parameters(modules):
    parameters = []
    seen_parameter_ids = set()

    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter.requires_grad and parameter_id not in seen_parameter_ids:
                parameters.append(parameter)
                seen_parameter_ids.add(parameter_id)

    return parameters


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


class DrivingPlannerModelB(nn.Module):
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
    ):
        super().__init__()
        self.future_steps = future_steps
        self.future_features = future_features
        self.normalize_camera = normalize_camera

        self.visual_backbone, visual_feature_dim = _build_resnet18_backbone(pretrained=pretrained_backbone)
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

        self.register_buffer(
            "camera_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "camera_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for parameter in self.visual_backbone.parameters():
            parameter.requires_grad = False

    def _prepare_camera(self, camera):
        camera = camera.float()

        # The existing dataset interface is unchanged, so normalize inside the model.
        if camera.detach().amax().item() > 1.5:
            camera = camera / 255.0

        if self.normalize_camera:
            camera = (camera - self.camera_mean) / self.camera_std

        return camera

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
        backbone_parameters = _collect_trainable_parameters([self.visual_backbone])
        head_parameters = _collect_trainable_parameters(head_modules)

        if backbone_learning_rate is None and backbone_lr_scale is None:
            return [
                {
                    "name": "main",
                    "params": head_parameters + backbone_parameters,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            ]

        resolved_backbone_lr = (
            backbone_learning_rate if backbone_learning_rate is not None else learning_rate * backbone_lr_scale
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


MODEL_REGISTRY = {
    'baseline': DrivingPlanner,
    'model_a': DrivingPlannerModelA,
    'model_b': DrivingPlannerModelB,
}


def build_model(name='baseline', **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_cls(**kwargs)
