"""VideoMAE ViT-B Model Wrapper for 82-Class Sports Action Recognition.

Adapts MCG-NJU/videomae-base-finetuned-kinetics with:
- Custom 82-class linear classifier head with Dropout
- Gradient checkpointing for 60-70% activation VRAM reduction
- Layer-wise learning rate decay parameter grouping (e.g., decay=0.75)
- Support for frozen backbone vs partial vs full fine-tuning
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification


class SportsVideoMAE(nn.Module):
    """Fine-Tuning Wrapper for VideoMAE on Sports Action Recognition."""

    def __init__(
        self,
        pretrained_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        num_classes: int = 82,
        dropout: float = 0.5,
        gradient_checkpointing: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_name = pretrained_name

        # Load pre-trained model config with updated num_labels
        self.config = VideoMAEConfig.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            id2label={i: f"class_{i}" for i in range(num_classes)},
            label2id={f"class_{i}": i for i in range(num_classes)},
            ignore_mismatched_sizes=True,
        )

        # Load backbone with pre-trained weights
        self.model = VideoMAEForVideoClassification.from_pretrained(
            pretrained_name,
            config=self.config,
            ignore_mismatched_sizes=True,
        )

        # Update classifier head with dropout
        hidden_dim = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Enable gradient checkpointing for VRAM memory savings
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final classification head."""
        for param in self.model.videomae.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_k_layers(self, k: int = 3) -> None:
        """Unfreeze the top K encoder transformer blocks + classifier head."""
        self.freeze_backbone()
        total_layers = len(self.model.videomae.encoder.layer)
        layers_to_unfreeze = self.model.videomae.encoder.layer[max(0, total_layers - k) :]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def get_parameter_groups_with_decay(
        self,
        base_lr: float,
        weight_decay: float = 0.05,
        layer_decay: float = 0.75,
    ) -> List[Dict]:
        """Construct layer-wise learning rate decay parameter groups for AdamW."""
        num_layers = len(self.model.videomae.encoder.layer)
        # Scales for each layer (lower layers get smaller LR)
        layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

        param_groups = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            no_decay = any(nd in name for nd in ["bias", "LayerNorm", "layernorm"])
            this_weight_decay = 0.0 if no_decay else weight_decay

            # Determine layer scale
            if "videomae.embeddings" in name:
                scale = layer_scales[0]
            elif "videomae.encoder.layer" in name:
                layer_idx = int(name.split("videomae.encoder.layer.")[1].split(".")[0])
                scale = layer_scales[layer_idx + 1]
            else:
                # Classifier head receives full LR
                scale = 1.0

            group_key = (scale, this_weight_decay)
            if group_key not in param_groups:
                param_groups[group_key] = {
                    "params": [],
                    "lr": base_lr * scale,
                    "weight_decay": this_weight_decay,
                }
            param_groups[group_key]["params"].append(param)

        return list(param_groups.values())

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        Args:
            pixel_values: Tensor of shape (B, T, C, H, W)
            labels: Optional Tensor of shape (B,)
        Returns:
            Dict containing 'loss' (if labels provided) and 'logits'
        """
        output = self.model(pixel_values=pixel_values, labels=labels, return_dict=True)
        return {
            "loss": output.loss,
            "logits": output.logits,
        }
