import torch
import torch.nn as nn
import json
from taming.models.vqgan import VQModel

class VQGAN(nn.Module):
    def __init__(self, config_path, ckpt_path):
        super().__init__()
        self.model = self.load_model(config_path, ckpt_path)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.quantize = self.model.quantize
        self.model.eval()

    def load_model(self, config_path, ckpt_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)

        model = VQModel(
            ddconfig={
                "double_z": False,
                "z_channels": cfg["z_channels"],
                "resolution": cfg["resolution"],
                "in_channels": cfg["num_channels"],
                "out_ch": cfg["num_channels"],
                "ch": cfg["hidden_channels"],
                "ch_mult": cfg["channel_mult"],
                "num_res_blocks": cfg["num_res_blocks"],
                "attn_resolutions": cfg["attn_resolutions"],
                "dropout": cfg["dropout"]
            },
            lossconfig={
                "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                "params": {
                    "disc_conditional": False,
                    "disc_in_channels": 3,
                    "disc_start": 0,
                    "codebook_weight": 1.0,
                    "pixelloss_weight": 1.0
                }
            },
            n_embed=cfg["num_embeddings"],
            embed_dim=cfg["quantized_embed_dim"],
            ckpt_path=None
        )

        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print("[\u2713] VQGAN weights loaded")
        return model

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        quant_out = self.quantize(z)
        return quant_out  # Return full tuple (z_q, z, indices)

    @torch.no_grad()
    def decode(self, tokens):
        z_q = self.quantize.get_codebook_entry(tokens)
        return self.decoder(z_q)


# === File: utils/token_helper.py ===
import torch

def create_interleaved_tokens(zin_list, zout_list=None):
    sequence = []
    for i, zin in enumerate(zin_list):
        zin_tensor = zin[2] if isinstance(zin, tuple) else zin  # Extract indices
        zout_tensor = zout_list[i][2] if zout_list and isinstance(zout_list[i], tuple) else zout_list[i]

        zin_flat = zin_tensor.view(-1)
        if zout_list is None:
            sequence.append(zin_flat)
        else:
            zout_flat = zout_tensor.view(-1)
            interleaved = torch.empty(zin_flat.numel() * 2, dtype=zin_flat.dtype, device=zin_flat.device)
            interleaved[0::2] = zin_flat
            interleaved[1::2] = zout_flat
            sequence.append(interleaved)
    return torch.cat(sequence, dim=0)

