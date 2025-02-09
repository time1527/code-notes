from dataclasses import dataclass, field
from typing import Dict, Optional
import os


# data
@dataclass
class DataConfig:
    path = "./data"
    n_classes: int = 10
    img_height: int = 32
    img_width: int = 32
    channels: int = 3


# model
@dataclass
class ModelConfig:
    n_layers: int = 6
    d_model: int = 128
    d_ffn: int = 512
    n_heads: int = 8
    dropout: float = 0.05

    img_height: int = DataConfig.img_height
    img_width: int = DataConfig.img_width
    channels: int = DataConfig.channels
    n_classes: int = DataConfig.n_classes

    patch_size: int = 4


# train
@dataclass
class TrainConfig:
    n_epochs: int = 100
    batch_size: int = 2048
    grad_clip: float = 1.0
    save_epochs: int = 50
    lr_config: Dict[str, float] = field(
        default_factory=lambda: {
            "lr": 1e-3,
            "warmup_iters": 100,
            "min_lr": 1e-5,
        }
    )
