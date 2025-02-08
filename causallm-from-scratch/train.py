import os
import time
import math
import warnings
import argparse
from contextlib import nullcontext

import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import PretrainDataset, SFTDataset
from log import setup_logger
from model import CausalLM
from tokenizer import SP_Tokenizer
from config import ModelConfig, PretrainConfig, SFTConfig, TokenizerConfig

warnings.filterwarnings("ignore")


def setup(train_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据任务选择初始化模型的方式
    if train_config.task_name == "pretrain":
        model = CausalLM(ModelConfig)
    elif train_config.task_name == "sft":
        checkpoint_dict = torch.load(train_config.pt_path, map_location=device)
        gptconf = checkpoint_dict["model_args"]
        model = CausalLM(gptconf)
        state_dict = checkpoint_dict["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)

    optimizer = model.configure_optimizers(
        learning_rate=train_config.lr_config["lr"],
        weight_decay=train_config.weight_decay,
        betas=train_config.betas,
        device_type=device,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(train_config.dtype in ["float16", "bfloat16"])
    )

    if train_config.task_name == "pretrain":
        train_dataset = PretrainDataset(
            data_path=train_config.train_data_bin,
            max_length=ModelConfig.max_seq_len,
        )
    elif train_config.task_name == "sft":
        train_dataset = SFTDataset(
            data_path=train_config.train_data_bin,
            max_length=ModelConfig.max_seq_len,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, pin_memory=True
    )

    tokenizer = SP_Tokenizer(TokenizerConfig.save_path)
    pad_id = tokenizer.pad_token_id
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    return model, scaler, optimizer, train_loader, loss_fn, pad_id


def get_lr(it, config, total_iters):
    """
    先线性增加，再余弦衰减
    0 -> warmup_iters：线性增加到lr
    warmup_iters -> lr_decay_iters：余弦衰减到min_lr
    lr_decay_iters -> max_iters：保持min_lr不变
    """
    lr = config["lr"]
    warmup_iters = config["warmup_iters"]
    lr_decay_iters = total_iters
    min_lr = config["min_lr"]

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


def iter_batches(dl, device):
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yield x, y


def train(config):
    # 初始化一些参数
    logger = setup_logger(config.task_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_config = config.lr_config
    grad_clip = config.grad_clip
    accumulation_steps = config.accumulation_steps
    log_interval = config.log_interval
    save_interval = config.save_interval
    total_iters = config.max_iters

    # 初始化模型、优化器、数据加载器等
    model, scaler, optimizer, train_loader, loss_fn, pad_id = setup(config)
    model.to(device)
    model.train()

    # 混合精度训练
    ctx = nullcontext() if device == "cpu" else torch.cuda.amp.autocast()

    # 初始化数据iter
    batch_iter = iter_batches(train_loader, device)
    x, y = next(batch_iter)

    logger.info(f"training config:{config}")
    logger.info(f"model config:{model.config}")

    os.makedirs("saves", exist_ok=True)

    losses = []
    iters_cnt = 0
    start_time = time.time()

    while iters_cnt < total_iters:
        # 学习率
        lr = get_lr(iters_cnt, lr_config, total_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            logits = model(x, y)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            # https://huggingface.co/blog/gradient_accumulation
            # loss除以非pad_id的token数量
            mask = (y != pad_id).float()
            num_valid_tokens = mask.sum()
            loss /= num_valid_tokens
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            loss /= accumulation_steps

        x, y = next(batch_iter)
        scaler.scale(loss).backward()

        if (iters_cnt + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (iters_cnt + 1) % log_interval == 0:
            spend_time = time.time() - start_time
            logger.info(
                "Iterations:[{}/{}] loss:{:.3f} lr:{:.7f} time:{:.2f}min".format(
                    iters_cnt + 1,
                    total_iters,
                    loss.item() * accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    int(spend_time // 60),
                )
            )
            losses.append(loss.item() * accumulation_steps)
            start_time = time.time()

        if (iters_cnt + 1) % save_interval == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model.config,
                "iter_num": iters_cnt,
                "config": config,
            }
            torch.save(checkpoint, f"saves/{config.task_name}-{iters_cnt}.pt")

        iters_cnt += 1

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model.config,
        "iter_num": iters_cnt,
        "config": config,
    }
    torch.save(checkpoint, f"saves/{config.task_name}-{total_iters}.pt")
    # loss曲线
    pd.DataFrame(losses).to_csv(f"saves/{config.task_name}-loss.csv", index=False)


def pretrain():
    pretrain_config = PretrainConfig()
    setup(pretrain_config)
    train(pretrain_config)


def sft():
    sft_config = SFTConfig()
    setup(sft_config)
    train(sft_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pretrain")
    args = parser.parse_args()
    if args.task == "pretrain":
        pretrain()
    elif args.task == "sft":
        sft()
    else:
        raise NotImplementedError
