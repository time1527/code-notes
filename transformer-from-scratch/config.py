from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    src_language: str = "de"
    tgt_language: str = "en"

    special_symbols: List[str] = field(
        default_factory=lambda: ["<unk>", "<pad>", "<bos>", "<eos>"]
    )
    unk_idx: int = 1
    pad_idx: int = 2
    bos_idx: int = 3
    eos_idx: int = 4


@dataclass
class ModelConfig:
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    d_model: int = 512
    d_ffn: int = 512
    n_heads: int = 8
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 25
    lr: float = 0.0001
