import torch
from safetensors.torch import save_file

bin_path = ".../RME/llama/pytorch_model.bin"

safetensor_path = ".../RME/llama/model.safetensors"

print("Loading .bin weights directly using torch.load...")
state_dict = torch.load(bin_path)  

print(f"Saving to {safetensor_path} as .safetensors...")
save_file(state_dict, safetensor_path)

print("Conversion complete.")
