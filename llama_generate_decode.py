import os
import torch
import numpy as np
import json
from transformers import AutoConfig, AutoModelForCausalLM
from taming.models.vqgan import VQModel
from PIL import Image
import torchvision.transforms as T

llama_dir = ".../RME/llama"
vqgan_ckpt = ".../RME/models/pytorch_model.bin"
vqgan_config = ".../RME/models/config.json"
token_input_path = ".../RME/llama_input/prompt_sequence.npy"
output_save_path = ".../RME/predicted_pathloss.png"

print("Loading LLaMA model...")
config = AutoConfig.from_pretrained(llama_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(llama_dir, config=config, trust_remote_code=True)
model.cuda().eval()

print("Loading VQ-GAN decoder...")
with open(vqgan_config, "r") as f:
    vq_cfg = json.load(f)
print("Loaded VQ-GAN config:", vq_cfg)

vqgan = VQModel(
    ddconfig={
        "z_channels": vq_cfg["z_channels"],
        "resolution": vq_cfg["resolution"],
        "in_channels": vq_cfg["num_channels"],
        "out_ch": vq_cfg["num_channels"],
        "ch": vq_cfg["hidden_channels"],
        "ch_mult": vq_cfg["channel_mult"],
        "num_res_blocks": vq_cfg["num_res_blocks"],
        "attn_resolutions": vq_cfg["attn_resolutions"],
        "dropout": vq_cfg["dropout"],
        "resamp_with_conv": vq_cfg["resample_with_conv"],
        "no_mid_attn": vq_cfg["no_attn_mid_block"],
    },
    lossconfig={"target": "torch.nn.Identity"},
    n_embed=vq_cfg["num_embeddings"],
    embed_dim=vq_cfg["quantized_embed_dim"]
)
vqgan.cuda().eval()

print("Working with z of shape (1, {}, 16, 16) = {} dimensions.".format(
    vq_cfg["z_channels"], vq_cfg["z_channels"] * 16 * 16))

print(f"Loading weights from: {vqgan_ckpt}")
state_dict = torch.load(vqgan_ckpt, map_location="cpu")
if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
vqgan.load_state_dict(state_dict, strict=False)

print("Loading input token sequence...")
input_tokens = np.load(token_input_path)
input_tokens = torch.tensor(input_tokens).unsqueeze(0).cuda()  # (1, T)

print("Generating token output...")
with torch.no_grad():
    generated = model.generate(
        input_tokens,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=0
    )

pred_tokens = generated[:, input_tokens.shape[1]:]
print("Generated token shape:", pred_tokens.shape)

assert pred_tokens.shape[1] == 256, f"Expected 256 tokens, got {pred_tokens.shape[1]}"

print("Decoding tokens to image...")
pred_tokens = pred_tokens[0].detach().cpu().numpy().reshape(16, 16)
pred_tokens = torch.tensor(pred_tokens).unsqueeze(0).long().cuda()  # (1, 16, 16)

with torch.no_grad():
    decoded = vqgan.decode(pred_tokens)  # (1, 3, H, W)

decoded = decoded.squeeze().cpu().clamp(0, 1)  # (3, H, W)
decoded_img = T.ToPILImage()(decoded)
decoded_img = decoded_img.resize((256, 256), Image.BICUBIC)
decoded_img.save(output_save_path)
print(f"Saved predicted image to: {output_save_path}")
