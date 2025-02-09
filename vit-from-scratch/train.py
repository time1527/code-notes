import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms, datasets

import os
import math
import json
from tqdm import tqdm
import multiprocessing

from config import DataConfig, ModelConfig, TrainConfig
from model import ViT
from log import setup_logger

logger = setup_logger()


def get_lr(it, config, total_iters):
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


def setup_data():
    num_workers = min(8, multiprocessing.cpu_count())

    train_dataset = datasets.CIFAR10(
        root=DataConfig.path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(
                    (DataConfig.img_height, DataConfig.img_width), padding=4
                ),
                transforms.Resize((DataConfig.img_height, DataConfig.img_width)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        ),
    )
    test_dataset = datasets.CIFAR10(
        root=DataConfig.path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((DataConfig.img_height, DataConfig.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        ),
    )
    logger.info(f"train dataset size: {len(train_dataset)}")
    logger.info(f"test dataset size: {len(test_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader


def setup():
    # 数据
    train_dataloader, test_dataloader = setup_data()
    # 模型
    model = ViT(ModelConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 优化器
    lr_config = TrainConfig().lr_config
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_config["lr"])
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    return train_dataloader, test_dataloader, model, optimizer, loss_fn


def test(model, loss_fn, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    correct = 0
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(
            tqdm(test_dataloader, desc="Testing")
        ):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()

            num_samples += images.size(0)

    avg_loss = total_loss / num_samples
    avg_accuracy = correct / num_samples

    return avg_loss, avg_accuracy


def train():
    train_dataloader, test_dataloader, model, optimizer, loss_fn = setup()
    logger.info(ModelConfig())
    logger.info(TrainConfig())
    logger.info("Start training...")

    lr_config = TrainConfig().lr_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    total_iters = len(train_dataloader) * TrainConfig.n_epochs
    current_iters = 0

    for epoch in range(TrainConfig.n_epochs):
        total_loss = 0
        correct = 0
        num_samples = 0
        model.train()
        for batch_idx, (images, labels) in enumerate(
            tqdm(train_dataloader, desc="Training")
        ):
            current_iters += 1

            lr = get_lr(current_iters, lr_config, total_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, labels)

            loss.backward()
            if TrainConfig.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), TrainConfig.grad_clip
                )
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            num_samples += images.size(0)

        train_loss = total_loss / num_samples
        train_acc = correct / num_samples
        test_loss, test_acc = test(model, loss_fn, test_dataloader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        logger.info(
            f"Epoch {epoch+1}/{TrainConfig.n_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )
        if (epoch + 1) % TrainConfig.save_epochs == 0:
            os.makedirs("saves", exist_ok=True)
            torch.save(model.state_dict(), f"saves/vit_epoch{epoch+1}.pth")

    os.makedirs("saves", exist_ok=True)
    torch.save(model.state_dict(), "saves/vit.pth")
    data = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
    }

    # 保存到文件
    with open("saves/results.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    train()
