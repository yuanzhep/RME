# import torch
# import torch.nn as nn
# import json
# from transformers import LlamaForCausalLM, LlamaConfig


# class LLaMAAutoregressiveModel(nn.Module):
#     """
#     Wrapper for pretrained LLaMA for autoregressive inference.
#     """

#     def __init__(self, config_path, ckpt_path):
#         super().__init__()
#         self.config = self.load_config(config_path)
#         self.model = LlamaForCausalLM(self.config)
#         self.load_weights(ckpt_path)
#         self.model.eval()

#     def load_config(self, config_path):
#         with open(config_path, "r") as f:
#             raw_cfg = json.load(f)
#         return LlamaConfig(**raw_cfg)

#     def load_weights(self, ckpt_path):
#         state_dict = torch.load(ckpt_path, map_location="cpu")
#         missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
#         print(f"[✓] LLaMA model loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")

#     @torch.no_grad()
#     def forward(self, input_ids, attention_mask=None):
#         return self.model(input_ids=input_ids, attention_mask=attention_mask)

#     @torch.no_grad()
#     def generate(self, input_ids, temperature=1.0, top_k=50, max_tokens=256):
#         """
#         Generate new tokens autoregressively from the input sequence.
#         """
#         return self.model.generate(
#             input_ids=input_ids,
#             temperature=temperature,
#             top_k=top_k,
#             do_sample=True,
#             max_new_tokens=max_tokens,
#         )


import torch
import torch.nn as nn
import json
from transformers import LlamaForCausalLM, LlamaConfig

class LLaMAAutoregressiveModel(nn.Module):
    def __init__(self, config_path, ckpt_path):
        super().__init__()
        self.config = self.load_config(config_path)
        self.model = LlamaForCausalLM(self.config)
        self.load_weights(ckpt_path)
        self.model.eval()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            raw_cfg = json.load(f)
        return LlamaConfig(**raw_cfg)

    def load_weights(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        print("[\u2713] LLaMA model loaded")

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256):
        return self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False)
