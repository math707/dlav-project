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


MODEL_REGISTRY = {
    'baseline': DrivingPlanner,
    'model_a': DrivingPlannerModelA,
}


def build_model(name='baseline', **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return model_cls(**kwargs)
