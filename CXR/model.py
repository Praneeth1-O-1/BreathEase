"""
Chest X-ray classification model with configurable backbone support.

Supported Backbones (Step 7):
  - "densenet121"    : CheXNet baseline (1024-dim feature vector)
  - "efficientnet_b4": Modern efficient CNN (1792-dim feature vector + 0.3 dropout)

The backbone is selected via config.BACKBONE or passed explicitly to CXRModel.
get_features() exposes the embedding vector (pre-classifier) for future multimodal fusion.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


class CXRModel(nn.Module):
    """
    Multi-label chest X-ray classifier supporting DenseNet121 and EfficientNet-B4.

    Parameters
    ----------
    backbone_name : str
        Name of the backbone ("densenet121" or "efficientnet_b4"). Default: config.BACKBONE.
    num_classes : int
        Number of output disease logits. Default: 14 (NIH disease labels).
    pretrained : bool
        Load ImageNet pretrained weights. Set False when loading a saved checkpoint.
    dropout_p : float
        Dropout probability in classifier head (used for EfficientNet-B4). Default: 0.3.
    """

    def __init__(
        self,
        backbone_name: str = config.BACKBONE,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        self.backbone_name = backbone_name.lower().strip()
        self.num_classes   = num_classes

        if self.backbone_name == "densenet121":
            weights  = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.densenet121(weights=weights)

            self.features = backbone.features
            self.avgpool  = nn.AdaptiveAvgPool2d(1)
            in_features   = backbone.classifier.in_features  # 1024

            self.classifier = nn.Linear(in_features, num_classes)

        elif self.backbone_name == "efficientnet_b4":
            weights  = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)

            self.features = backbone.features
            self.avgpool  = backbone.avgpool
            in_features   = backbone.classifier[1].in_features  # 1792

            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. "
                f"Choose either 'densenet121' or 'efficientnet_b4'."
            )

        self.in_features = in_features

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, H, W)

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Raw logits. BCEWithLogitsLoss handles sigmoid internally.
        """
        x        = self.features(x)
        x        = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits   = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embedding vector before classifier head.
        Returns (B, 1024) for DenseNet121, (B, 1792) for EfficientNet-B4.
        """
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)