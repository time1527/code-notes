from dataclasses import dataclass, field
from typing import Dict, Optional
import os

# 数据
data_dir = "/root/autodl-tmp/"


# tokenizer
@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    # unk_token = "<unk>"
    # bos_token = "<s>"
    # eos_token = "</s>"
    # pad_token = "<pad>"
    # unk_token_id: int = 0
    # bos_token_id: int = 1
    # eos_token_id: int = 2
    pad_token_id: int = 3
    train_file = os.path.join(data_dir, "TinyStoriesV2_cleaned/data/train")
    save_path = f"sptok{vocab_size}.model"


# model
@dataclass
class ModelConfig:
    n_layers: int = 8
    d_model: int = 288
    d_ffn: int = 768
    n_heads: int = 8
    n_kv_heads: Optional[int] = 4
    dropout: float = 0.0
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    base: int = 10000
    vocab_size: int = TokenizerConfig.vocab_size


# pretrain
@dataclass
class PretrainConfig:
    task_name = "pretrain"
    train_data_path = os.path.join(data_dir, "TinyStoriesV2_cleaned/data/train")
    train_data_bin = os.path.join(data_dir, "pretrain_data_bin")
    max_iters: int = 10000
    # batch_size x accumulation_steps x max_seq_len
    # 256 x 4 x 256 ~200,000
    batch_size: int = 256
    weight_decay: float = 1e-1

    lr_config: Dict[str, float] = field(
        default_factory=lambda: {
            "lr": 5e-4,
            "warmup_iters": 1000,
            "min_lr": 0.0,
        }
    )

    grad_clip: float = 1.0
    accumulation_steps: int = 4
    dtype = "bfloat16"
    log_interval: int = 10
    save_interval: int = 1000
    betas = (0.9, 0.95)


# sft
@dataclass
class SFTConfig:
    task_name = "sft"
    train_data_path = os.path.join(data_dir, "TinyStoriesInstruct")
    train_data_bin = os.path.join(data_dir, "sft_data_bin")
    pt_path = f"/root/autodl-tmp/llama/saves/pretrain-{PretrainConfig.max_iters}.pt"
    max_iters: int = 10000
    # batch_size x accumulation_steps x max_seq_len
    #  ~200,000
    batch_size: int = 256
    weight_decay: float = 1e-1

    lr_config: Dict[str, float] = field(
        default_factory=lambda: {
            "lr": 5e-4,
            "warmup_iters": 1000,
            "min_lr": 0.0,
        }
    )

    grad_clip: float = 1.0
    accumulation_steps: int = 4
    dtype = "bfloat16"
    log_interval: int = 10
    save_interval: int = 1000
    betas = (0.9, 0.95)
