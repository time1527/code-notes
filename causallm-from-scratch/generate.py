import torch
import argparse
from model import CausalLM
from tokenizer import SP_Tokenizer
from config import ModelConfig, TokenizerConfig
from dataset import SFT_TEMPLETE


def output(args):
    t = args.t  # temperature
    n = args.n  # max length
    top_k = args.top_k  # top k sampling
    rp = args.rp
    prompt = args.prompt  # prompt
    checkpoint = args.path  # checkpoint path
    if args.template == SFT_TEMPLETE:  # template
        prompt = SFT_TEMPLETE.format(instruct=prompt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SP_Tokenizer(TokenizerConfig.save_path)

    # checkpoint加载
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = checkpoint_dict["model_args"]
    model = CausalLM(gptconf)
    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 将字符串分词，得到input_ids
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        out = model.generate(
            idx=x,
            max_new_tokens=n - len(input_ids),
            temperature=t,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=rp,
        )
        print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="One day, Lily came into")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=ModelConfig.max_seq_len)
    parser.add_argument("--rp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--path", type=str, default="/root/autodl-tmp/llama/saves/pretrain-3999.pt"
    )
    parser.add_argument("--template", type=str, default=SFT_TEMPLETE)
    args = parser.parse_args()

    output(args)
