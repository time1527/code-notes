import os
from tqdm import tqdm
from typing import List
import sentencepiece as spm
from config import TokenizerConfig


class SP_Tokenizer:
    def __init__(self, tokenizer_model):
        model_path = tokenizer_model
        assert os.path.isfile(model_path), model_path
        self.sp_model = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.vocab_size: int = self.sp_model.vocab_size()
        self.bos_token_id: int = self.sp_model.bos_id()
        self.eos_token_id: int = self.sp_model.eos_id()
        self.pad_token_id: int = self.sp_model.pad_id()

        self.bos_token = self.sp_model.id_to_piece(self.bos_token_id)
        self.eos_token = self.sp_model.id_to_piece(self.eos_token_id)

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """string -> input_ids"""
        assert type(s) is str
        t = self.sp_model.encode(s)
        if add_bos:
            t = [self.bos_token_id] + t
        if add_eos:
            t = t + [self.eos_token_id]
        return t

    def decode(self, t: List[int]) -> str:
        """input_ids -> string"""
        return self.sp_model.decode(t)


def train_sptok(tokenizer_config: TokenizerConfig):
    """使用sentencepiece训练分词器"""

    # 训练数据处理
    def preprocess(file, output_file):
        if os.path.isdir(file):
            file_paths = []
            for root, dirs, files in os.walk(file):
                for f in files:
                    file_path = os.path.join(root, f)
                    file_paths.append(file_path)
        elif os.path.isfile(file):
            file_paths = [file]
        else:
            raise ValueError(f"file {file} is not a valid file or directory")

        with open(output_file, "w", encoding="utf-8") as out_f:
            for file_path in tqdm(file_paths):
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        line = line.replace("<|endoftext|>", "")
                        if line:
                            out_f.write(line + "\n")

    raw = tokenizer_config.train_file
    txt_file = os.path.join(os.path.dirname(raw), "tokenizer_train_tmp.txt")
    preprocess(raw, txt_file)
    prefix = ".".join(tokenizer_config.save_path.split(".")[:-1])

    # 训练sentencepiece模型
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    # https://github.com/yanqiangmiffy/how-to-train-tokenizer/tree/main
    # https://github.com/baichuan-inc/Baichuan-7B/tree/main?tab=readme-ov-file#%E5%88%86%E8%AF%8D
    # https://zhuanlan.zhihu.com/p/655281268
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/issues/409
    # https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/issues/358
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(
        input=txt_file,
        input_format="text",
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=tokenizer_config.vocab_size,
        character_coverage=1,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        unk_surface=r" \342\201\207 ",
        pad_id=tokenizer_config.pad_token_id,
    )

    # 删除临时文件
    os.remove(txt_file)
    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


if __name__ == "__main__":
    train_sptok(TokenizerConfig)
    sp = SP_Tokenizer(TokenizerConfig.save_path)
    print(sp.sp_model.id_to_piece(sp.pad_token_id))
