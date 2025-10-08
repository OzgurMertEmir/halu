from halu.features.build import build_features_df
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_openllm():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    tok.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype="auto", device_map="auto", trust_remote_code=True, attn_implementation="eager"
    ).eval()
    df = build_features_df(model, tok, "openllm", size=20, seed=1337)

    print(df.head())
    print(df.columns)
    print(df.info())
    print(df.describe())

