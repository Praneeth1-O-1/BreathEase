import sys
import torch
import torch.nn as nn

sys.path.append("pytorch")

from models import Cnn14


class AudioModel(nn.Module):

    def __init__(self, checkpoint_path=None, freeze_backbone=False):
        super().__init__()

        # -------------------------------------------------
        # CNN14 Backbone
        # -------------------------------------------------
        self.backbone = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527
        )

        # -------------------------------------------------
        # Load pretrained weights
        # -------------------------------------------------
        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu"
            )
            self.backbone.load_state_dict(checkpoint["model"])

        # -------------------------------------------------
        # Optionally freeze CNN14 for a stable head-only fine-tuning baseline.
        # -------------------------------------------------
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        # -------------------------------------------------
        # Temporal Attention
        # -------------------------------------------------

        # -------------------------------------------------
        # Projection Layer
        # -------------------------------------------------
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # -------------------------------------------------
        # Residual Block
        # -------------------------------------------------
        self.residual = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512)
        )

        # -------------------------------------------------
        # Final Classifier
        # -------------------------------------------------
        self.classifier = nn.Linear(512, 2)

        # -------------------------------------------------
        # Parameter Count
        # -------------------------------------------------
        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        print(f"Trainable Parameters : {trainable:,}")
        print(f"Total Parameters     : {total:,}")

    def forward(self, x):

        # -------------------------------------------------
        # CNN14 Feature Extraction
        # -------------------------------------------------
        outputs = self.backbone(
            x,
            return_features=True
        )

        feature_map = outputs["feature_map"]      # (B,2048,T)

        # (B,T,2048)
        feature_map = feature_map.transpose(1, 2)

        # Mean pooling across time
        context = feature_map.mean(dim=1)

        x = self.projection(context)

        x = self.residual(x)

        logits = self.classifier(x)

        return logits, None
