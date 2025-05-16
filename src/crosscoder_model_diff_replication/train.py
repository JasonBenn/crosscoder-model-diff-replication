import torch
from transformers import AutoModelForCausalLM
from pathlib import Path
from trainer import Trainer
from buffer import Buffer, PrecachedActivationsBuffer
from utils import (
    HookedTransformer,
    arg_parse_update_cfg,
    load_tokens,
)
import argparse
DEVICE = "cuda:0"

default_cfg = {
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    # "d_in": base_model.cfg.d_model,
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
    "tiny_mode": False,
    "use_cached_activations": True,
}

cfg = arg_parse_update_cfg(default_cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cached_activations", action="store_true")
    parser.add_argument("--tiny_mode", action="store_true")
    args = parser.parse_args()

    cfg['tiny_mode'] = args.tiny_mode

    if args.use_cached_activations:
        all_tokens = load_tokens("mixed")

        base_model = HookedTransformer.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            hf_model=AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct"
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

        cfg["d_in"] = base_model.cfg.d_model

        buffer = Buffer(cfg, base_model, chat_model, all_tokens)
    else:
        activations_dir = Path("/mnt/polished-lake/data/model-diffing/") 
        suffix = 'tiny' if args.tiny_mode else 'full'
        base_act_path = activations_dir / f"activations_base_{suffix}.pt" / "partition_0" / "0_activations.mm"
        ft_act_path = activations_dir / f"activations_ft_{suffix}.pt" / "partition_0" / "0_activations.mm"

        cfg["d_in"] = 128  # TODO: ?

        buffer = PrecachedActivationsBuffer(base_act_path, ft_act_path, cfg)

    trainer = Trainer(cfg, buffer)
    trainer.train()
