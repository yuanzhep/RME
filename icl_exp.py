import os
import torch
import torchvision.transforms as T
from PIL import Image
from models.vqgan import VQGAN
from models.llama_wrapper import LLaMAAutoregressiveModel
from utils.token_helper import create_interleaved_tokens

DEVICE = "cuda"
VQGAN_CONFIG = "models/config.json"
VQGAN_WEIGHTS = "models/pytorch_model.bin"
LLAMA_CONFIG = "models/llama300m_config.json"
LLAMA_WEIGHTS = "models/llama300m.bin"
INPUT_DIR = "data0/input"
OUTPUT_DIR = "data0/output"
QUERY_NAME = "B25_Ant1_f1_S47"
PROMPT_NAMES = [
    "B25_Ant1_f1_S40",
    "B25_Ant1_f1_S45"
]

vqgan = VQGAN(VQGAN_CONFIG, VQGAN_WEIGHTS).to(DEVICE).eval()
llama = LLaMAAutoregressiveModel(LLAMA_CONFIG, LLAMA_WEIGHTS).to(DEVICE).eval()

def load_tensor_from_png(path, to_gray=False):
    img = Image.open(path)
    img = img.convert("L" if to_gray else "RGB")
    img = img.resize((256, 256))
    tensor = T.ToTensor()(img).unsqueeze(0).to(DEVICE)  # shape: [1, C, 256, 256]
    return tensor

zin_list, zout_list = [], []
for name in PROMPT_NAMES:
    pI = load_tensor_from_png(os.path.join(INPUT_DIR, name + ".png"), to_gray=False)
    pO = load_tensor_from_png(os.path.join(OUTPUT_DIR, name + ".png"), to_gray=True)
    pO = pO.repeat(1, 3, 1, 1)  # repeat grayscale to 3 channels

    print(f"[→] Encoding {name}: input shape={pI.shape}, output shape={pO.shape}")

    zin_out = vqgan.encode(pI)
    zout_out = vqgan.encode(pO)
    zin = zin_out[2] if isinstance(zin_out, tuple) else zin_out
    zout = zout_out[2] if isinstance(zout_out, tuple) else zout_out
    zin_list.append(zin)
    zout_list.append(zout)

sequence = create_interleaved_tokens(zin_list, zout_list)

query_tensor = load_tensor_from_png(os.path.join(INPUT_DIR, QUERY_NAME + ".png"), to_gray=False)
z_query_out = vqgan.encode(query_tensor)
z_query_in = z_query_out[2] if isinstance(z_query_out, tuple) else z_query_out
z_query_in = z_query_in.view(-1)  # shape: [256]

final_sequence = torch.cat([sequence, z_query_in], dim=0).unsqueeze(0)  # shape: [1, T]

with torch.no_grad():
    prediction = llama.generate(input_ids=final_sequence, gen_len=256)
    z_pred = prediction[:, -256:]  # shape: [1, 256]

z_pred = z_pred.view(1, 16, 16)  # reshape to grid
recon = vqgan.decode(z_pred).squeeze(0).clamp(0, 1).cpu()  # [3, 256, 256]
recon_img = T.ToPILImage()(recon)

os.makedirs("predictions", exist_ok=True)
save_path = os.path.join("predictions", f"pred_{QUERY_NAME}.png")
recon_img.save(save_path)
print(f"Prediction saved to {save_path}")
