# %%
import torch
import os
from transformers import AutoModelForCausalLM

from src.crosscoder_model_diff_replication.crosscoder import CrossCoder
from transformer_lens import HookedTransformer
import torch
import einops
import plotly.express as px

DEVICE = "cuda:0"

# %%
torch.set_grad_enabled(False)  # important for memory
# TODO: this directory should be specific to a (base_model, rl_model) tuple
coder = CrossCoder.load(version=6, checkpoint_version=3)
norms = coder.W_dec.norm(dim=-1)
relative_norms = norms[:, 1] / norms.sum(dim=-1)
print(relative_norms.shape, norms.shape)

# %%
import plotly.io as pio
# for classic Jupyter Notebook
pio.renderers.default = "notebook"

fig = px.histogram(
    relative_norms.detach().cpu().numpy(),
    title="R1 Distilled vs Post-training on Math",
    labels={"value": "Relative decoder norm strength"},
    nbins=200,
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents")

# Update x-axis ticks
fig.update_xaxes(
    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
    ticktext=['0', '0.25', '0.5', '0.75', '1.0']
)
fig.show()
# %%
cross_coder = coder
shared_latent_mask = (relative_norms < 0.7) & (relative_norms > 0.3)
cosine_sims = (cross_coder.W_dec[:, 0, :] * cross_coder.W_dec[:, 1, :]).sum(dim=-1) / (cross_coder.W_dec[:, 0, :].norm(dim=-1) * cross_coder.W_dec[:, 1, :].norm(dim=-1))
print(cosine_sims.shape, shared_latent_mask.shape)
fig = px.histogram(
    cosine_sims[shared_latent_mask].to(torch.float32).detach().cpu().numpy(),
    #title="Cosine similarity of decoder vectors between models",
    log_y=True,  # Sets the y-axis to log scale
    range_x=[-1, 1],  # Sets the x-axis range from -1 to 1
    nbins=100,  # Adjust this value to change the number of bins
    labels={"value": "Cosine similarity of decoder vectors between models"}
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents (log scale)")

fig.show()


# %%
from pathlib import Path
ROOT_DIR = Path(__file__).parent
CODE_DIR = ROOT_DIR.parent

# import argparse 
# argparse = argparse.ArgumentParser()
# argparse.add_argument("--tokens_path", type=str)
# args = argparse.parse_args()

# if not args or not os.path.exists(args.tokens_path):
#     print(f"Tokens path {args.tokens_path} does not exist, exiting...")
#     from utils import load_tokens
#     all_tokens = load_tokens("mixed")
#     torch.save(all_tokens, args.tokens_path)
# else:
#     all_tokens = torch.load(args.tokens_path)

all_tokens = torch.load(CODE_DIR / "upweight-reason/data/tokenized/mbpp__1/tokens.pt")
all_tokens.shape

# %%
import copy
folded_cross_coder = copy.deepcopy(cross_coder)


def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    cross_coder.W_enc.data[0, :, :] = cross_coder.W_enc.data[0, :, :] * base_scaling_factor
    cross_coder.W_enc.data[1, :, :] = cross_coder.W_enc.data[1, :, :] * chat_scaling_factor

    cross_coder.W_dec.data[:, 0, :] = cross_coder.W_dec.data[:, 0, :] / base_scaling_factor
    cross_coder.W_dec.data[:, 1, :] = cross_coder.W_dec.data[:, 1, :] / chat_scaling_factor

    cross_coder.b_dec.data[0, :] = cross_coder.b_dec.data[0, :] / base_scaling_factor
    cross_coder.b_dec.data[1, :] = cross_coder.b_dec.data[1, :] / chat_scaling_factor
    return cross_coder

base_estimated_scaling_factor = 0.2758961493232058
chat_estimated_scaling_factor = 0.24422852496546169
folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_estimated_scaling_factor, chat_estimated_scaling_factor)
folded_cross_coder = folded_cross_coder.to(torch.bfloat16)
device = 'cuda:0'
torch.set_grad_enabled(False) # important for memory
from transformers import AutoModelForCausalLM

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    hf_model=AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    device=device,
    dtype=torch.bfloat16
)

chat_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    hf_model=AutoModelForCausalLM.from_pretrained("aokellermann/deepscaler_1.5b_16k_eurus_2_math"),
    device=device,
    dtype=torch.bfloat16
)
from functools import partial

def splice_act_hook(act, hook, spliced_act):
    act[:, 1:, :] = spliced_act # Drop BOS
    return act

def zero_ablation_hook(act, hook):
    act[:] = 0
    return act

def get_ce_recovered_metrics(tokens, model_A, model_B, cross_coder):
    # get clean loss
    ce_clean_A = model_A(tokens, return_type="loss")
    ce_clean_B = model_B(tokens, return_type="loss")

    # get zero abl loss
    ce_zero_abl_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )
    ce_zero_abl_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )

    # bunch of annoying set up for splicing
    _, cache_A = model_A.run_with_cache(
        tokens,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
        )
    resid_act_A = cache_A[cross_coder.cfg["hook_point"]]

    _, cache_B = model_B.run_with_cache(
        tokens,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
        )
    resid_act_B = cache_B[cross_coder.cfg["hook_point"]]

    cross_coder_input = torch.stack([resid_act_A, resid_act_B], dim=0)
    cross_coder_input = cross_coder_input[:, :, 1:, :] # Drop BOS
    cross_coder_input = einops.rearrange(
        cross_coder_input,
        "n_models batch seq_len d_model -> (batch seq_len) n_models d_model",
    )

    cross_coder_output = cross_coder.decode(cross_coder.encode(cross_coder_input))
    cross_coder_output = einops.rearrange(
        cross_coder_output,
        "(batch seq_len) n_models d_model -> n_models batch seq_len d_model", batch = tokens.shape[0]
    )
    cross_coder_output_A = cross_coder_output[0]
    cross_coder_output_B = cross_coder_output[1]

    # get spliced loss
    ce_loss_spliced_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_A))],
    )
    ce_loss_spliced_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks = [(cross_coder.cfg["hook_point"], partial(splice_act_hook, spliced_act=cross_coder_output_B))],
    )

    # compute % CE recovered metric
    ce_recovered_A = 1 - ((ce_loss_spliced_A - ce_clean_A) / (ce_zero_abl_A - ce_clean_A))
    ce_recovered_B = 1 - ((ce_loss_spliced_B - ce_clean_B) / (ce_zero_abl_B - ce_clean_B))

    metrics = {
        "ce_loss_spliced_A": ce_loss_spliced_A.item(),
        "ce_loss_spliced_B": ce_loss_spliced_B.item(),
        "ce_clean_A": ce_clean_A.item(),
        "ce_clean_B": ce_clean_B.item(),
        "ce_zero_abl_A": ce_zero_abl_A.item(),
        "ce_zero_abl_B": ce_zero_abl_B.item(),
        "ce_diff_A": (ce_loss_spliced_A - ce_clean_A).item(),
        "ce_diff_B": (ce_loss_spliced_B - ce_clean_B).item(),
        "ce_recovered_A": ce_recovered_A.item(),
        "ce_recovered_B": ce_recovered_B.item(),
    }
    return metrics

tokens = all_tokens[torch.randperm(len(all_tokens))[:1]]
ce_metrics = get_ce_recovered_metrics(tokens, base_model, chat_model, folded_cross_coder)
ce_metrics
base_estimated_scaling_factor = 0.2835
chat_estimated_scaling_factor = 0.2533

# %%
import copy
folded_cross_coder = copy.deepcopy(cross_coder)

def fold_activation_scaling_factor(cross_coder, base_scaling_factor, chat_scaling_factor):
    cross_coder.W_enc.data[0, :, :] = cross_coder.W_enc.data[0, :, :] * base_scaling_factor
    cross_coder.W_enc.data[1, :, :] = cross_coder.W_enc.data[1, :, :] * chat_scaling_factor

    # cross_coder.W_dec.data[:, 0, :] = cross_coder.W_dec.data[:, 0, :] / base_scaling_factor
    # cross_coder.W_dec.data[:, 1, :] = cross_coder.W_dec.data[:, 1, :] / chat_scaling_factor

    # cross_coder.b_dec.data[0, :] = cross_coder.b_dec.data[0, :] / base_scaling_factor
    # cross_coder.b_dec.data[1, :] = cross_coder.b_dec.data[1, :] / chat_scaling_factor
    return cross_coder

folded_cross_coder = fold_activation_scaling_factor(folded_cross_coder, base_estimated_scaling_factor, chat_estimated_scaling_factor)
from sae_vis.model_fns import CrossCoderConfig, CrossCoder

encoder_cfg = CrossCoderConfig(d_in=base_model.cfg.d_model, d_hidden=cross_coder.cfg["dict_size"], apply_b_dec_to_input=False)
sae_vis_cross_coder = CrossCoder(encoder_cfg)
sae_vis_cross_coder.load_state_dict(folded_cross_coder.state_dict())
sae_vis_cross_coder = sae_vis_cross_coder.to("cuda:0")
sae_vis_cross_coder = sae_vis_cross_coder.to(torch.bfloat16)
import numpy as np

N_SAMPLED_IDXS = 1000

rl_idxs = [651, 9474, 11316, 13604, 10125, 3022, 10617, 14207, 8822, 14979, 7607, 10997, 6812, 14650, 6216, 6036, 2496, 15914, 10179, 2732]
base_idxs = [4806, 11550, 12624, 4141, 12214, 9677, 15103, 2351, 8040, 4249, 3352, 6912, 5845, 12164, 489, 9723, 1136, 11801, 10012, 8448]
common_idxs = [13317, 2054, 15366, 5126, 12, 14349, 8206, 6159, 4111, 11280]

all_idxs = np.arange(0, 16384)
exclusions = set(rl_idxs) | set(base_idxs)
allowed = np.array([i for i in all_idxs if i not in exclusions], dtype=int)
test_feature_idx = np.random.choice(allowed, size=N_SAMPLED_IDXS, replace=False).tolist()

# %%
from sae_vis.data_config_classes import SaeVisConfig
sae_vis_config = SaeVisConfig(
    hook_point = folded_cross_coder.cfg["hook_point"],
    features = test_feature_idx,
    verbose = True,
    minibatch_size_tokens=4,
    minibatch_size_features=16,
)
from sae_vis.data_storing_fns import SaeVisData
sae_vis_data = SaeVisData.create(
    encoder = sae_vis_cross_coder,
    encoder_B = None,
    model_A = base_model,
    model_B = chat_model,
    tokens = all_tokens[:128], # in practice, better to use more data
    cfg = sae_vis_config,
)
def relative_decoder_strength_diff(feature_idx):
    a, b = sae_vis_data.feature_data_dict[feature_idx].feature_tables_data.relative_decoder_strength_values
    return b - a

rl_idxs = []
base_idxs = []

for feature_idx in test_feature_idx:
    diff = relative_decoder_strength_diff(feature_idx)
    if diff > 0.85:
        rl_idxs.append(feature_idx)
    elif diff < -0.85:
        base_idxs.append(feature_idx)

normal_idxs = list(set(test_feature_idx) - set(rl_idxs) - set(base_idxs))[:10]

print("rl_idxs =", rl_idxs)
print("base_idxs =", base_idxs)
print("common_idxs =", normal_idxs)

min_len = min([len(rl_idxs), len(base_idxs)])  # don't care about common_idxs as much
min_len
final_indexes = rl_idxs[:min_len] + base_idxs[:min_len] + normal_idxs[:min_len]

# %%
from sae_vis.data_config_classes import SaeVisConfig
sae_vis_config = SaeVisConfig(
    hook_point = folded_cross_coder.cfg["hook_point"],
    features = final_indexes,
    verbose = True,
    minibatch_size_tokens=4,
    minibatch_size_features=16,
)
from sae_vis.data_storing_fns import SaeVisData
sae_vis_data = SaeVisData.create(
    encoder = sae_vis_cross_coder,
    encoder_B = None,
    model_A = base_model,
    model_B = chat_model,
    tokens = all_tokens[:128], # in practice, better to use more data
    cfg = sae_vis_config,
)
filename = ROOT_DIR / f"visualizations/crosscoder_v{6}__checkpoint_{3}__128_tokens.html"
sae_vis_data.save_feature_centric_vis(filename)
folded_cross_coder.cfg["hook_point"]
rl_logits, rl_cache = chat_model.run_with_cache(
    tokens,
    names_filter=cross_coder.cfg["hook_point"],
    return_type='logits',
)
print(rl_logits.shape, rl_cache)
tokens

# %%
