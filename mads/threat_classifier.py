import torch
import torch.nn as nn


class ThreatClassifier(nn.Module):
    """Binary neural network classifier for threat detection.

    Architecture: 5 → 32 → 16 → 1 (raw logit)
    Use with BCEWithLogitsLoss during training and torch.sigmoid() at inference.
    """

    def __init__(self, input_dim: int = 5):
        super().__init__()
        # 737 trainable parameters
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(16, 1),   # raw logit — BCEWithLogitsLoss handles sigmoid
        # )

        # 424 trainable parameters
        self.net = nn.Sequential(
            nn.Linear(input_dim, 9),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(9, 1)  # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
