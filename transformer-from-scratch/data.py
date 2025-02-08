import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k


class TranslationDataset(Dataset):
    def __init__(self, split, data_config):
        self.data_config = data_config
        self.split = split
        self.data = list(
            Multi30k(
                split=split,
                language_pair=(data_config.src_language, data_config.tgt_language),
            )
        )

        self.token_transform = {
            data_config.src_language: get_tokenizer(
                "spacy", language="de_core_news_sm"
            ),
            data_config.tgt_language: get_tokenizer("spacy", language="en_core_web_sm"),
        }

        self.vocab_transform = {}
        for ln in [data_config.src_language, data_config.tgt_language]:
            train_iter = Multi30k(
                split="train",
                language_pair=(data_config.src_language, data_config.tgt_language),
            )
            self.vocab_transform[ln] = build_vocab_from_iterator(
                self._yield_tokens(train_iter, ln),
                min_freq=1,
                specials=data_config.special_symbols,
                special_first=True,
            )
            self.vocab_transform[ln].set_default_index(data_config.unk_idx)

        self.src_vocab_size = len(self.vocab_transform[data_config.src_language])
        self.tgt_vocab_size = len(self.vocab_transform[data_config.tgt_language])

        self.text_transform = {}
        for ln in [data_config.src_language, data_config.tgt_language]:
            self.text_transform[ln] = self._sequential_transforms(
                self.token_transform[ln],
                self.vocab_transform[ln],
                self._tensor_transform,
            )

    def _yield_tokens(self, data_iter, language):
        language_index = {
            self.data_config.src_language: 0,
            self.data_config.tgt_language: 1,
        }
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])

    def _sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def _tensor_transform(self, token_ids):
        return torch.cat(
            (
                torch.tensor([self.data_config.bos_idx]),
                torch.tensor(token_ids),
                torch.tensor([self.data_config.eos_idx]),
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sample, tgt_sample = self.data[idx]
        src_tensor = self.text_transform[self.data_config.src_language](
            src_sample.rstrip("\n")
        )
        tgt_tensor = self.text_transform[self.data_config.tgt_language](
            tgt_sample.rstrip("\n")
        )
        return src_tensor, tgt_tensor


class TranslationDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, pad_idx):
        super().__init__(dataset, batch_size=batch_size, collate_fn=self.collate_fn)
        self.pad_idx = pad_idx

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx)
        return src_batch, tgt_batch


# # Usage
# if __name__ == "__main__":
#     from config import DataConfig

#     data_config = DataConfig()
#     train_dataset = TranslationDataset("valid", data_config)
#     train_dataloader = TranslationDataLoader(
#         train_dataset, batch_size=32, pad_idx=data_config.pad_idx
#     )

#     print(train_dataset[0])
#     print(train_dataset.src_vocab_size)
