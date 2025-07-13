import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from vqgan_legacy import VQModel  
import torchvision.transforms as T

vqgan_dir = ".../RME/models"
config_path = os.path.join(vqgan_dir, "config.json")
ckpt_path = os.path.join(vqgan_dir, "pytorch_model.bin")

token_dir = ".../tokens"
vis_dir = ".../token_vis"
recon_dir = ".../recon"
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(recon_dir, exist_ok=True)

print("Loading VQ-GAN...")
with open(config_path, "r") as f:
    config = json.load(f)

model = VQModel(
    resolution=config["resolution"],
    in_channels=config["num_channels"],
    out_ch=config["num_channels"],
    ch=config["hidden_channels"],
    z_channels=config["z_channels"],
    num_res_blocks=config["num_res_blocks"],
    attn_resolutions=config["attn_resolutions"],
    dropout=config["dropout"],
    resamp_with_conv=config["resample_with_conv"],
    in_channels_aux=None,
    ch_mult=config["channel_mult"],
    double_z=False,
    use_mid_attn=not config["no_attn_mid_block"]
)

sd = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(sd, strict=False)
model.eval()
print("VQ-GAN loaded.")

token_paths = sorted(glob(os.path.join(token_dir, "*.npy")))
for token_path in token_paths:
    img_name = os.path.splitext(os.path.basename(token_path))[0].replace("_tokens", "")
    print(f"\nProcessing: {img_name}")

    tokens = np.load(token_path)  # [h, w]
    h, w = tokens.shape
    unique_tokens = np.unique(tokens)
    vocab_size = model.quantize.n_e if hasattr(model.quantize, "n_e") else 8192

    print(f"  → Unique tokens used: {len(unique_tokens)} / {vocab_size}")

    # ======== Visualize Token Map ========
    plt.figure(figsize=(6, 6))
    plt.imshow(tokens, cmap="tab20")
    plt.title(f"{img_name} Tokens ({len(unique_tokens)} used)")
    plt.axis("off")
    pdf_path = os.path.join(vis_dir, f"{img_name}_tokens.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  → Token map saved as PDF: {pdf_path}")

    tokens_torch = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, h, w]
    with torch.no_grad():
        z = model.quantize.get_codebook_entry(tokens_torch, shape=None)  # [1, h, w, c]
        z = z.permute(0, 3, 1, 2).contiguous()  # [1, c, h, w]
        recon = model.decode(z)  # [1, C, H, W]

        recon_img = recon.squeeze().permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        recon_img = (recon_img + 1.0) / 2.0  # [-1, 1] → [0, 1]
        recon_img = np.clip(recon_img, 0, 1)

    recon_path = os.path.join(recon_dir, f"{img_name}_recon.png")
    plt.imsave(recon_path, recon_img)
    print(f"  → Recon image saved: {recon_path}")
