import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):

        # x
        # (Batch, Time, Features)

        scores = self.score(x)

        weights = torch.softmax(scores, dim=1)

        context = torch.sum(weights * x, dim=1)

        return context, weights