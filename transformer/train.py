import torch
import torch.nn as nn

from timeit import default_timer as timer

from model import Transformer
from data import TranslationDataLoader, TranslationDataset
from config import DataConfig, ModelConfig, TrainConfig

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TranslationDataset("train", DataConfig())
valid_dataset = TranslationDataset("valid", DataConfig())


################################ train ####################################################################
def train_epoch(model, optimizer):
    model.train()
    losses = 0

    train_dataloader = TranslationDataLoader(
        train_dataset, batch_size=TrainConfig.batch_size, pad_idx=DataConfig.pad_idx
    )

    for src, tgt in train_dataloader:
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_input)

        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        mask = (
            tgt_input != DataConfig.pad_idx
        ).float()  # 生成一个 mask，标记非 pad_id 的位置为 1，pad_id 的位置为 0
        num_valid_tokens = mask.sum()
        loss /= num_valid_tokens

        loss.backward()
        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = TranslationDataLoader(
        valid_dataset, batch_size=TrainConfig.batch_size, pad_idx=DataConfig.pad_idx
    )

    for src, tgt in val_dataloader:
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        mask = (
            tgt_input != DataConfig.pad_idx
        ).float()  # 生成一个 mask，标记非 pad_id 的位置为 1，pad_id 的位置为 0
        num_valid_tokens = mask.sum()
        loss /= num_valid_tokens

        losses += loss.item()

    return losses / len(list(val_dataloader))


def greedy_decode(model, src, max_len, start_symbol):
    src = src.to(DEVICE)
    enc_output = model.encoder(src)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        enc_output = enc_output.to(DEVICE)
        out = model.decoder(ys, src, enc_output)
        prob = model.projection(out)
        _, prob = prob.squeeze(0).max(dim=-1, keepdim=False)
        # prob = history + next token prediction
        next_word = prob[-1].item()  # i.e. prob[i].item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == DataConfig.eos_idx:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = train_dataset.text_transform[DataConfig.src_language](src_sentence).view(
        1, -1
    )
    num_tokens = src.shape[1]
    tgt_tokens = greedy_decode(
        model, src, max_len=num_tokens + 5, start_symbol=DataConfig.bos_idx
    ).flatten()
    return (
        " ".join(
            train_dataset.vocab_transform[DataConfig.tgt_language].lookup_tokens(
                list(tgt_tokens.cpu().numpy())
            )
        )
        .replace("<bos>", "")
        .replace("<eos>", "")
    )


if __name__ == "__main__":
    transformer = Transformer(
        train_dataset.src_vocab_size,
        train_dataset.tgt_vocab_size,
        ModelConfig.n_encoder_layers,
        ModelConfig.n_decoder_layers,
        ModelConfig.d_model,
        ModelConfig.d_ffn,
        ModelConfig.n_heads,
        ModelConfig.dropout,
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=DataConfig.pad_idx, reduction="sum"
    )

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=TrainConfig.lr, betas=(0.9, 0.98), eps=1e-9
    )

    for epoch in range(1, TrainConfig.num_epochs + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print(
            (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"
            )
        )

    print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
