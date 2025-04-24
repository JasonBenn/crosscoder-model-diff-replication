import torch
from transformers import AutoModelForCausalLM

from crosscoder import CrossCoder
from trainer import Trainer
from utils import (
    HookedTransformer,
    arg_parse_update_cfg,
    load_sft_reasoning_tokens,
)
import torch
from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download, notebook_login
import json
import einops
import plotly.express as px

from typing import NamedTuple

DEVICE = "cuda:0"
torch.set_grad_enabled(False)  # important for memory
coder = CrossCoder.load(5, 2)

# all_tokens = load_sft_reasoning_tokens()

# base_model = HookedTransformer.from_pretrained(
#     "Qwen/Qwen2.5-1.5B",
#     hf_model=AutoModelForCausalLM.from_pretrained(
#         "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#     ),
#     device=DEVICE,
#     dtype=torch.bfloat16,
# )

# chat_model = HookedTransformer.from_pretrained(
#     "Qwen/Qwen2.5-1.5B",
#     hf_model=AutoModelForCausalLM.from_pretrained(
#         "aokellermann/deepscaler_1.5b_16k_eurus_2_math"
#     ),
#     device=DEVICE,
#     dtype=torch.bfloat16,
# )
