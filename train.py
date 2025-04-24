import torch
from transformers import AutoModelForCausalLM

from trainer import Trainer
from utils import (
    HookedTransformer,
    arg_parse_update_cfg,
    load_sft_reasoning_tokens,
)

DEVICE = "cuda:0"
all_tokens = load_sft_reasoning_tokens()

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    hf_model=AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ),
    device=DEVICE,
    dtype=torch.bfloat16,
)

chat_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    hf_model=AutoModelForCausalLM.from_pretrained(
        "aokellermann/deepscaler_1.5b_16k_eurus_2_math"
    ),
    device=DEVICE,
    dtype=torch.bfloat16,
)

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "deepscaler_1.5b_16k_eurus_2_math",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 4,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "crosscoders",
    "wandb_entity": "jasoncbenn",
}

cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
