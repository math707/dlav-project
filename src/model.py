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
