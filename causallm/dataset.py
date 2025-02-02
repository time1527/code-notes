import os
import re
import glob
import tqdm
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from tokenizer import SP_Tokenizer
from config import ModelConfig

SFT_TEMPLETE = """Human: {instruct}\nAssistant: """


def process_shard_for_pretrain(args, train_config, tokenizer_config):
    """
    处理单个shard的数据，将文本数据转换为tokenized的二进制文件。

    Args:
        args: shard_id, shard
        train_config: 训练配置
        tokenizer_config: 分词器配置

    Returns:
        None
    """
    # shard_id：数字index
    # shard：文件名
    shard_id, shard = args

    # 加载分词器
    tokenizer = SP_Tokenizer(tokenizer_config.save_path)

    # 读取数据
    with open(shard, "r", encoding="utf-8") as file:
        data = [line.replace("<|endoftext|>", "").strip() for line in file]

    # 分词
    all_tokens = []
    for text in tqdm(data, position=shard_id):
        text = text.strip()
        # 与llama2.c不同之处：add_eos=True
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        all_tokens.extend(tokens)

    # 转换成np.array
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # 得到存放的文件名
    bin_dir = train_config.train_data_bin
    shard_basename = os.path.basename(shard)
    bin_basename = shard_basename.replace(".txt", ".bin")
    tokenized_filename = os.path.join(bin_dir, bin_basename)

    # 写入文件
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())

    # 计算平均长度，根据bos_token
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def extract_field(keyword, record, keywords):
    """
    通用函数，用于根据关键词提取字段内容。

    Args:
        keyword (str): 需要提取的关键词。
        record (str): 输入的文本记录。
        keywords (list): 所有可能的关键词列表，用于确定字段的结束边界。

    Returns:
        str: 提取到的字段内容（去掉多余空格），如果关键词不存在，返回 None。
    """
    # 动态生成下一个字段的边界，所有其他关键词
    other_keywords = [kw for kw in keywords if kw != keyword]
    # 使用负向前瞻(?!)来确保字段提取时不跨越到下一个字段
    boundary = "|".join(map(re.escape, other_keywords))  # 转义特殊字符

    # 生成正则表达式：以 `keyword:` 开头，直到下一个字段或文本结尾
    pattern = rf"{keyword}:\s*(.*?)(?=\s*(?:{boundary})\s*:|$)"

    # 查找并返回匹配内容
    match = re.search(pattern, record, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_keywords_and_items(text):
    """
    按记录分割文本并提取关键词字段，进行后处理。

    Args:
        text (str): 输入的长文本，包含多条记录，用 <|endoftext|> 分隔。

    Returns:
        list: 包含每条记录的字段提取结果的字典列表。
    """
    # 定义关键词列表
    keywords = ["Story", "Features", "Summary", "Words", "Random sentence"]

    # 按记录分割
    records = text.split("<|endoftext|>")

    # print(f"Total records: {len(records)}")  # 打印记录总数

    data = []
    for record in records:
        if not record.strip():
            continue  # 跳过空记录

        # 提取各个字段
        story = extract_field("Story", record, keywords)
        if not story or not story.strip():
            continue  # 如果 Story 不存在或为空，跳过当前记录

        features = extract_field("Features", record, keywords)
        summary = extract_field("Summary", record, keywords)
        words = extract_field("Words", record, keywords)
        sentence = extract_field("Random sentence", record, keywords)

        # 后处理逻辑
        result = {"Story": story}  # Story 一定存在，因此直接添加

        # 添加非空字段到字典
        if features:
            result["Features"] = [f.strip() for f in features.split(",") if f.strip()]
        if summary and summary.strip():
            result["Summary"] = summary
        if words:
            result["Words"] = [w.strip() for w in words.split(",") if w.strip()]
        if sentence and sentence.strip():
            result["Sentence"] = sentence

        # 添加到结果列表
        data.append(result)

    return data


def generate_sentence_based_on_item(item):
    """
    根据给定的 item 动态生成基于现有字段的故事生成提示。

    Args:
        item (dict): 包含关键词的字典，可能包括 'Words', 'Sentence', 'Features', 'Summary' 等字段。

    Returns:
        str: 根据现有字段随机生成的故事提示。
    """
    # 定义关键词及其模板
    templates = {
        "Words": [
            """Using the words [{words}], craft a compelling story that revolves around them.""",
            """Write a story that prominently features these words: [{words}].""",
            """Create a story where the following words play a central role: [{words}].""",
            """Build a narrative incorporating these key words: [{words}].""",
            """Generate a story in which these words shape the plot: [{words}].""",
        ],
        "Sentence": [
            """Write a story that integrates the sentence: "{sentence}". Make it a meaningful part of the plot.""",
            """Create a story where this sentence is pivotal: "{sentence}".""",
            """Develop a story that naturally includes this sentence: "{sentence}".""",
            """Using the sentence "{sentence}", craft a story with a strong narrative structure.""",
            """Ensure the story revolves around the impact or context of this sentence: "{sentence}".""",
        ],
        "Features": [
            """Create a story that incorporates the following features: [{features}]. Bring them to life in an engaging way.""",
            """Write a story that highlights these features: [{features}]. Make them integral to the narrative.""",
            """Craft a story where these features shape the plot: [{features}].""",
            """Develop a story showcasing the following elements: [{features}].""",
            """Using these features—[{features}]—construct a meaningful story.""",
        ],
        "Summary": [
            """Expand this summary into a full story: "{summary}". Add rich details and bring it to life.""",
            """Turn the following summary into a captivating story: "{summary}".""",
            """Using this summary as inspiration, write a detailed story: "{summary}".""",
            """Develop a complete story based on the summary: "{summary}".""",
            """Build an engaging story that reflects this summary: "{summary}".""",
        ],
    }

    # 随机选择存在的关键词
    available_keys = [key for key in templates.keys() if item.get(key)]
    if not available_keys:
        return "Please write a story for me."

    selected_key = random.choice(available_keys)

    # 准备替换字段的内容
    replacements = {
        "words": ", ".join(item.get("Words", [])),
        "sentence": item.get("Sentence", ""),
        "features": ", ".join(item.get("Features", [])),
        "summary": item.get("Summary", ""),
    }

    # 从模板中选择并格式化
    selected_template = random.choice(templates[selected_key])
    return selected_template.format(**replacements)


def process_shard_for_sft(args, train_config, tokenizer_config, batch_size=500000):
    """
    处理单个shard的数据，将文本数据转换为tokenized的二进制文件。
    Args:
        args: shard_id, shard
        train_config: 训练配置
        tokenizer_config: 分词器配置
        batch_size: 每个batch的大小，如果数据太大，可能一次处理不完（内存），需要分批处理
    Returns:
        None
    """
    # shard_id：数字index, shard：文件名
    shard_id, shard = args

    # 加载分词器
    tokenizer = SP_Tokenizer(tokenizer_config.save_path)

    # 最大长度，用于截断/补全
    max_length = ModelConfig.max_seq_len

    # 读取数据
    with open(shard, "r", encoding="utf-8") as file:
        data = [line.strip() for line in file]

    # 提取关键词
    all_text = "".join(data)
    data = extract_keywords_and_items(all_text)

    # 分批处理初始化
    all_tokens = []
    all_labels = []
    batch_counter = 0

    # 得到处理的文件名，用于生成存放的文件名
    bin_dir = train_config.train_data_bin
    shard_basename = os.path.basename(shard)

    # 分批处理、保存数据
    for item in tqdm(data, position=shard_id):
        output = item["Story"].strip()
        if not output:
            continue

        # 根据模板生成instruct，并分词
        instruct = SFT_TEMPLETE.format(instruct=generate_sentence_based_on_item(item))
        encode_instruct = tokenizer.encode(instruct, add_bos=True, add_eos=False)
        # 模型应该输出的内容，分词
        encode_output = tokenizer.encode(output, add_bos=False, add_eos=True)

        # 拼接
        encode_item = encode_instruct + encode_output
        label = [tokenizer.pad_token_id] * len(encode_instruct) + encode_output

        # 补全/截断
        if len(encode_item) < max_length:  # 补全
            encode_item += [tokenizer.pad_token_id] * (max_length - len(encode_item))
            label += [tokenizer.pad_token_id] * (max_length - len(label))
            assert len(encode_item) == len(label) == max_length
            all_tokens.append(encode_item)
            all_labels.append(label)
        elif (
            len(encode_instruct) < max_length // 2
        ):  # 截断：如果拼接的长度超过最大长度，但instruct不到最大长度的一半，则截断
            encode_item = encode_item[:max_length]
            label = label[:max_length]
            assert len(encode_item) == len(label) == max_length
            all_tokens.append(encode_item)
            all_labels.append(label)

        # 如果达到batch_size，则保存
        if len(all_tokens) >= batch_size:
            # 转换为np.array
            all_tokens = np.array(all_tokens, dtype=np.uint16)
            all_labels = np.array(all_labels, dtype=np.uint16)

            # 保存tokens
            token_bin_basename = shard_basename.replace(
                ".txt", f"tokens_{batch_counter}.bin"
            )
            token_filename = os.path.join(bin_dir, token_bin_basename)
            with open(token_filename, "wb") as f:
                f.write(all_tokens.tobytes())
            print(f"Saved {token_filename}")

            # 保存labels
            label_bin_basename = token_bin_basename.replace(
                f"tokens_{batch_counter}.bin", f"labels_{batch_counter}.bin"
            )
            label_filename = os.path.join(bin_dir, label_bin_basename)
            with open(label_filename, "wb") as f:
                f.write(all_labels.tobytes())
            print(f"Saved {label_filename}")

            # 重置，处理下一个batch
            all_tokens = []
            all_labels = []
            batch_counter += 1

    # 保存剩余的tokens和labels
    if all_tokens:
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        all_labels = np.array(all_labels, dtype=np.uint16)

        token_bin_basename = shard_basename.replace(
            ".txt", f"tokens_{batch_counter}.bin"
        )
        token_filename = os.path.join(bin_dir, token_bin_basename)
        with open(token_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {token_filename}")

        label_bin_basename = token_bin_basename.replace(
            f"tokens_{batch_counter}.bin", f"labels_{batch_counter}.bin"
        )
        label_filename = os.path.join(bin_dir, label_bin_basename)
        with open(label_filename, "wb") as f:
            f.write(all_labels.tobytes())
        print(f"Saved {label_filename}")


def pretokenize(train_config, tokenizer_config):
    """
    对原始数据进行预处理，将文本数据转换为tokenized的二进制文件。
    Args:
        train_config: 训练配置
        tokenizer_config: 分词器配置
    Returns:
        None
    """
    # 获取所有shard的文件名
    data_dir = train_config.train_data_path  # 原始数据路径
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    # 获取任务名
    task = train_config.task_name

    # 存放bin的数据路径，也是PretrainDataset/SFTDataset类的数据路径
    bin_dir = train_config.train_data_bin
    os.makedirs(bin_dir, exist_ok=True)

    # 根据任务名，选择处理函数
    if task == "pretrain":
        f = process_shard_for_pretrain
    elif task == "sft":
        f = process_shard_for_sft

    # 多进程处理
    fun = partial(
        f,
        train_config=train_config,
        tokenizer_config=tokenizer_config,
    )
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, max_length):
        super().__init__()
        self.bin_dir = data_path  # data_bin
        self.max_length = max_length

    def __iter__(self):
        seed = 42
        rng = random.Random(seed)
        shard_filenames = sorted(glob.glob(os.path.join(self.bin_dir, "*.bin")))

        while True:
            # 打乱索引->>遍历shard->>切分->>yield
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_length
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_length
                    end = start + self.max_length + 1
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class SFTDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, max_length):
        super().__init__()
        self.bin_dir = data_path  # data_bin
        self.max_length = max_length

    def __iter__(self):
        seed = 42
        rng = random.Random(seed)
        token_filenames = sorted(glob.glob(os.path.join(self.bin_dir, "*tokens_*.bin")))
        label_filenames = sorted(glob.glob(os.path.join(self.bin_dir, "*labels_*.bin")))

        ixs_list = list(range(len(token_filenames)))
        while True:
            # 随机打乱索引->>遍历shard->>切分->>yield
            # 与PretrainDataset不同之处在于，这里的x和y分别来自tokens和labels
            # 因为instruct部分是不计算损失函数的
            # 对应于：process_shard_for_sft使用pad_token_id补全
            # 对应于：交叉熵损失函数ignore_index=tokenizer.pad_token_id
            rng.shuffle(ixs_list)
            for i in ixs_list:
                m = np.memmap(token_filenames[i], dtype=np.uint16, mode="r")
                label_m = np.memmap(label_filenames[i], dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_length
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_length
                    end = start + self.max_length + 1
                    chunk_x = torch.from_numpy((m[start:end]).astype(np.int64))
                    chunk_y = torch.from_numpy((label_m[start:end]).astype(np.int64))
                    x = chunk_x[:-1]
                    y = chunk_y[1:]
                    yield x, y


if __name__ == "__main__":
    from config import PretrainConfig, TokenizerConfig, SFTConfig
    from tokenizer import SP_Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pretrain")
    args = parser.parse_args()
    if args.task == "pretrain":
        pretokenize(PretrainConfig, TokenizerConfig)
    elif args.task == "sft":
        pretokenize(SFTConfig, TokenizerConfig)
    else:
        raise NotImplementedError
