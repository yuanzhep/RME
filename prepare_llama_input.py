import os
import numpy as np
from PIL import Image
import torch
from vqgan_legacy import VQModel
import torchvision.transforms as T
import json
from glob import glob

vqgan_dir = ".../RME/models"
config_path = os.path.join(vqgan_dir, "config.json")
ckpt_path = os.path.join(vqgan_dir, "pytorch_model.bin")
input_dir = ".../RME/data0/input"
output_dir = ".../RME/data0/output"
save_path = ".../RME/data0/llama_input/prompt_sequence.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

prompt_inputs = ["B25_Ant1_f1_S40.png", "B25_Ant1_f1_S45.png"]
prompt_outputs = ["B25_Ant1_f1_S40.png", "B25_Ant1_f1_S45.png"]
query_input = "B25_Ant1_f1_S47.png"

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

transform = T.Compose([
    T.Resize((config["resolution"], config["resolution"])),
    lambda x: x * 2. - 1.
])

def encode_image(img_path):
    img = Image.open(img_path)
    img_np = np.array(img)

    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 1:
        img_np = np.concatenate([img_np] * 3, axis=-1)

    img_tensor = torch.from_numpy(img_np).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = transform(img_tensor).unsqueeze(0)

    with torch.no_grad():
        z_output = model.encode(img_tensor)
        token_indices = z_output[2][2]
        return token_indices.cpu().numpy()

token_sequence = []

for in_img, out_img in zip(prompt_inputs, prompt_outputs):
    in_path = os.path.join(input_dir, in_img)
    out_path = os.path.join(output_dir, out_img)
    in_tokens = encode_image(in_path)
    out_tokens = encode_image(out_path)
    token_sequence.extend(in_tokens.tolist())
    token_sequence.extend(out_tokens.tolist())

query_path = os.path.join(input_dir, query_input)
query_tokens = encode_image(query_path)
token_sequence.extend(query_tokens.tolist())

np.save(save_path, np.array(token_sequence, dtype=np.int32))
print(f"Saved LLaMA input sequence to {save_path}")
