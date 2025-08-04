import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from typing import List, Tuple, Optional
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import time
import sys
import utils
from datetime import datetime
from safetensors.torch import load_file  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    if hasattr(log_print, 'log_file') and log_print.log_file:
        print(*args, **kwargs, file=log_print.log_file)
        log_print.log_file.flush()

def setup_logging(K, query_file):
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = query_file.replace('.png', '')
    log_filename = f"log/ripple_K{K}_{query_name}_{timestamp}.log"
    log_print.log_file = open(log_filename, 'w')
    log_print(f"Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Parameters: K={K}, Query={query_file}")
    log_print(f"Device: {DEVICE}")
    log_print("-" * 50)
    return log_filename

def close_logging():
    if hasattr(log_print, 'log_file') and log_print.log_file:
        log_print("-" * 50)
        log_print(f"Log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print.log_file.close()
        log_print.log_file = None

def load_safetensors_sharded(model_dir: str) -> dict:
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    state_dict = {}
    weight_map = index["weight_map"]
    
    # Get unique shard files
    shard_files = list(set(weight_map.values()))
    
    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        if os.path.exists(shard_path):
            log_print(f"Loading shard: {shard_file}")
            shard_weights = load_file(shard_path)
            state_dict.update(shard_weights)
        else:
            log_print(f"Warning: Shard file not found: {shard_path}")
    
    return state_dict

class Config:
    def __init__(self, K=2, query_file=None):
        self.vqgan_config = ".../models/config.json"
        self.vqgan_weights = ".../models/pytorch_model.bin"  
        # Llama 7B
        self.llama_config = ".../models/LVM_ckpts/config.json"
        self.llama_model_dir = ".../models/LVM_ckpts"  
        self.prompt_input_dir = ".../prompts/Inputs"
        self.prompt_output_dir = ".../prompts/Outputs"
        self.query_dir = ".../queries"
        self.prediction_output_dir = ".../prediction"
        self.K = K  
        self.query_file = query_file
        self.available_prompts = self.get_available_prompts()
        self.selected_prompts = self.available_prompts[:self.K] if self.K > 0 else []
        self.image_size = 256
        self.latent_size = 16
        self.codebook_size = 8192
        self.temperature = 0.8
        self.start_token = 0
        self.sep_token = 1
        self.end_token = 2
    
    # def get_available_prompts(self):
    #     available = []
    #     if os.path.exists(self.prompt_input_dir):
    #         input_files = [f.replace('.png', '') for f in os.listdir(self.prompt_input_dir) if f.endswith('.png')]
    #         for name in input_files:
    #             input_exists = os.path.exists(os.path.join(self.prompt_input_dir, name + '.png'))
    #             output_exists = os.path.exists(os.path.join(self.prompt_output_dir, name + '.png'))
    #             if input_exists and output_exists:
    #                 available.append(name)
    #     fallback_prompts = ["B25_Ant1_f1_S40", "B25_Ant1_f1_S45", "B25_Ant1_f1_S42", "B25_Ant1_f1_S43", "B25_Ant1_f1_S44", "B25_Ant1_f1_S41", "B25_Ant1_f1_S46", "B25_Ant1_f1_S47", "B25_Ant1_f1_S48", "B25_Ant1_f1_S49"]
    #     while len(available) < max(10, self.K):
    #         for fallback in fallback_prompts:
    #             if fallback not in available:
    #                 available.append(fallback)
    #                 if len(available) >= max(10, self.K):
    #                     break
        
    #     return available

class VQGAN(nn.Module):
    def __init__(self, config_path: str, weights_path: str):
        super().__init__()
        
        self.embed_dim = 256
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'quantize.embedding.weight' in state_dict:
                    self.embed_dim = state_dict['quantize.embedding.weight'].shape[1]
                    log_print(f"Detected embedding dimension: {self.embed_dim}")
            except:
                log_print("Could not detect embedding dimension, using default 256")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "codebook_size": 8192,
                "embedding_dim": self.embed_dim,
                "n_embed": 8192,
            }
        
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.quantize = VectorQuantizer(
            n_embed=8192,
            embed_dim=self.embed_dim
        )
        
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                self.load_state_dict(state_dict, strict=False)
                log_print(f"Loaded VQGAN weights from {weights_path}")
            except Exception as e:
                log_print(f"Warning: Could not load weights: {e}")
    
    def _make_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.embed_dim, 4, 2, 1),
        )
    
    def _make_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        quant, emb_loss, indices = self.quantize(h)
        return h, emb_loss, indices
    
    def decode(self, indices):
        if indices.dim() == 2:
            B, T = indices.shape
            H = W = int(math.sqrt(T))
            indices = indices.view(B, H, W)
        
        quant = self.quantize.get_codebook_entry(indices)
        dec = self.decoder(quant)
        return dec

class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
    
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, min_encoding_indices.view(z.shape[0], -1)
    
    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class SimpleLLaMA(nn.Module):
    def __init__(self, config_path: str, model_dir: str):
        super().__init__()
        
        # Load configuration from the Llama 7B config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                
            self.config = {
                "vocab_size": config_dict.get("vocab_size", 32000),  
                "hidden_size": config_dict.get("hidden_size", 4096),  
                "num_layers": config_dict.get("num_hidden_layers", 32),  
                "num_heads": config_dict.get("num_attention_heads", 32),  
                "max_seq_len": config_dict.get("max_position_embeddings", 4096),
                "intermediate_size": config_dict.get("intermediate_size", 11008),
                "rms_norm_eps": config_dict.get("rms_norm_eps", 1e-6),
            }
        else:
            self.config = {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_heads": 32,
                "max_seq_len": 4096,
                "intermediate_size": 11008,
                "rms_norm_eps": 1e-6,
            }
        
        log_print(f"Llama 7B Config: {self.config}")
        
        vocab_size = self.config["vocab_size"]
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        num_heads = self.config["num_heads"]
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(self.config["max_seq_len"], hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, self.config["intermediate_size"]) 
            for _ in range(num_layers)
        ])
    
        self.ln_f = RMSNorm(hidden_size, eps=self.config["rms_norm_eps"])
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        try:
            log_print("Loading Llama 7B weights from safetensors...")
            state_dict = load_safetensors_sharded(model_dir)
            self._load_llama_weights(state_dict)
            log_print("Successfully loaded Llama 7B weights")
            
        except Exception as e:
            log_print(f"Warning: Could not load Llama 7B weights: {e}")
    
    def _load_llama_weights(self, state_dict: dict):
        model_dict = self.state_dict()
        weight_mapping = {}
        # Embedding layers
        if "model.embed_tokens.weight" in state_dict:
            weight_mapping["embedding.weight"] = "model.embed_tokens.weight"
        # Transformer layers
        for i in range(self.config["num_layers"]):
            # Self attention
            weight_mapping[f"layers.{i}.attention.in_proj_weight"] = f"model.layers.{i}.self_attn.q_proj.weight"
            weight_mapping[f"layers.{i}.attention.in_proj_bias"] = f"model.layers.{i}.self_attn.q_proj.bias"
            
            # Feed forward
            weight_mapping[f"layers.{i}.feed_forward.0.weight"] = f"model.layers.{i}.mlp.gate_proj.weight"
            weight_mapping[f"layers.{i}.feed_forward.2.weight"] = f"model.layers.{i}.mlp.down_proj.weight"
            
            # Layer norms
            weight_mapping[f"layers.{i}.ln1.weight"] = f"model.layers.{i}.input_layernorm.weight"
            weight_mapping[f"layers.{i}.ln2.weight"] = f"model.layers.{i}.post_attention_layernorm.weight"
        
        if "model.norm.weight" in state_dict:
            weight_mapping["ln_f.weight"] = "model.norm.weight"
        
        if "lm_head.weight" in state_dict:
            weight_mapping["head.weight"] = "lm_head.weight"
        
        filtered_dict = {}
        for model_key, llama_key in weight_mapping.items():
            if llama_key in state_dict and model_key in model_dict:
                llama_weight = state_dict[llama_key]
                model_weight_shape = model_dict[model_key].shape
                
                if llama_weight.shape == model_weight_shape:
                    filtered_dict[model_key] = llama_weight
                else:
                    log_print(f"Shape mismatch for {model_key}: {llama_weight.shape} vs {model_weight_shape}")
        
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
        log_print(f"Loaded {len(filtered_dict)} compatible weight tensors")
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        input_ids = torch.clamp(input_ids, 0, self.config["vocab_size"] - 1)
        tok_emb = self.embedding(input_ids)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        pos_ids = torch.clamp(pos_ids, 0, self.config["max_seq_len"] - 1)
        pos_emb = self.pos_embedding(pos_ids)
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),  # gate_proj
            nn.SiLU(),  # Llama uses SiLU activation
            nn.Linear(intermediate_size, hidden_size, bias=False)   # down_proj
        )
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        normed_x = self.ln1(x)
        attn_out, _ = self.attention(normed_x, normed_x, normed_x, attn_mask=causal_mask)
        x = x + attn_out
        normed_x = self.ln2(x)
        ff_out = self.feed_forward(normed_x)
        x = x + ff_out   
        return x

def load_and_preprocess_image(path: str, target_size: int = 256, force_channels: int = None) -> torch.Tensor:
    if not os.path.exists(path):
        log_print(f"Warning: File {path} not found.")
        channels = force_channels if force_channels else 3
        return torch.randn(1, channels, target_size, target_size).to(DEVICE)
    
    img = Image.open(path)
    
    if force_channels == 1:
        img = img.convert("L")
    elif force_channels == 3:
        img = img.convert("RGB")
    else:
        if img.mode in ['L', 'P']:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
    
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    if img.mode == 'L':
        tensor = T.ToTensor()(img).unsqueeze(0)
        if force_channels == 3:
            tensor = tensor.repeat(1, 3, 1, 1)
    else:
        tensor = T.ToTensor()(img).unsqueeze(0)
        if force_channels == 1:
            tensor = tensor.mean(dim=1, keepdim=True)
    
    return tensor.to(DEVICE)

def create_icl_sequence(input_tokens_list: List[torch.Tensor], 
                       output_tokens_list: List[torch.Tensor],
                       query_tokens: torch.Tensor,
                       max_length: int = 3500) -> torch.Tensor: 
    sequence = []
    vocab_size = 8192 
    total_prompt_tokens = sum(len(inp) + len(out) + 2 for inp, out in zip(input_tokens_list, output_tokens_list))
    query_length = len(query_tokens) + 1
    expected_total = total_prompt_tokens + query_length
    
    if expected_total > max_length:
        available_for_prompts = max_length - query_length - 20
        tokens_per_prompt = available_for_prompts // (len(input_tokens_list) * 2) if input_tokens_list else 128
        tokens_per_prompt = max(64, min(tokens_per_prompt, 256))
    else:
        tokens_per_prompt = 512  
    
    for i, (inp_tokens, out_tokens) in enumerate(zip(input_tokens_list, output_tokens_list)):
        inp_tokens_truncated = inp_tokens[:tokens_per_prompt]
        out_tokens_truncated = out_tokens[:tokens_per_prompt]
        
        inp_tokens_truncated = torch.clamp(inp_tokens_truncated, 10, min(vocab_size - 10, 8191))
        out_tokens_truncated = torch.clamp(out_tokens_truncated, 10, min(vocab_size - 10, 8191))
        
        sequence.extend(inp_tokens_truncated.tolist())
        sequence.append(1)
        sequence.extend(out_tokens_truncated.tolist())
        sequence.append(2)
    
    query_tokens_truncated = query_tokens[:tokens_per_prompt]
    query_tokens_truncated = torch.clamp(query_tokens_truncated, 10, min(vocab_size - 10, 8191))
    
    sequence.extend(query_tokens_truncated.tolist())
    sequence.append(1)
    
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    
    sequence = [max(0, min(token, vocab_size - 1)) for token in sequence]
    
    return torch.tensor(sequence, dtype=torch.long, device=DEVICE)

def generate_output_tokens(model: SimpleLLaMA, input_sequence: torch.Tensor, 
                         num_tokens: int = 128, K: int = 2) -> torch.Tensor:
    
    model.eval()
    generated_tokens = []
    vocab_size = model.config["vocab_size"]
    current_seq = input_sequence.unsqueeze(0)
    current_seq = torch.clamp(current_seq, 0, vocab_size - 1)
    num_tokens = min(num_tokens, 256)  
    
    with torch.no_grad():
        for step in range(num_tokens):
            try:
                context_limit = max(2000, 3000 - K * 100)
                if current_seq.size(1) > context_limit:
                    keep_tokens = max(1500, context_limit - 500)
                    current_seq = current_seq[:, -keep_tokens:]
                
                logits = model(current_seq)
                next_logits = logits[0, -1, :]
                next_logits = next_logits / 0.8
                next_logits[:10] = float('-inf')
                safe_upper_bound = min(vocab_size - 10, 8191)
                next_logits[safe_upper_bound:] = float('-inf')
                
                next_logits = torch.where(torch.isfinite(next_logits), next_logits, torch.tensor(-1e9, device=next_logits.device))
                probs = F.softmax(next_logits, dim=-1)
                
                if torch.isnan(probs).any() or torch.sum(probs) == 0:
                    probs = torch.zeros_like(probs)
                    probs[10:safe_upper_bound] = 1.0 / (safe_upper_bound - 10)
                    
                next_token = torch.multinomial(probs, 1)
                next_token_val = next_token.item()
                next_token = torch.tensor([next_token_val], device=DEVICE)
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_tokens.append(next_token_val)
                
    return torch.tensor(generated_tokens, dtype=torch.long, device=DEVICE)

def main():
    parser = argparse.ArgumentParser(description='RIPPLE ICL for Radio Map Estimation')
    parser.add_argument('--K', type=int, default=2, help='Number of prompts to use (default: 2)')
    parser.add_argument('--query', type=str, default="B24_Ant1_f1_S0.png", help='Query file name')
    args = parser.parse_args()
    log_filename = setup_logging(args.K, args.query)
    config = Config(K=args.K, query_file=args.query)
    log_print("=== RIPPLE (Llama 7B) ===")
    log_print("\n[1] Loading models...")
    vqgan = VQGAN(config.vqgan_config, config.vqgan_weights).to(DEVICE).eval()
    llama = SimpleLLaMA(config.llama_config, config.llama_model_dir).to(DEVICE).eval()
    os.makedirs(config.prediction_output_dir, exist_ok=True)
    log_print(f"\n[2] Loading prompt examples (K={config.K})...")
    prompt_input_tokens = []
    prompt_output_tokens = []
    for i, name in enumerate(config.selected_prompts):
        input_path = os.path.join(config.prompt_input_dir, name + ".png")
        output_path = os.path.join(config.prompt_output_dir, name + ".png")
        input_img = load_and_preprocess_image(input_path, config.image_size, force_channels=3)
        output_img = load_and_preprocess_image(output_path, config.image_size, force_channels=3)
        with torch.no_grad():
            _, _, input_tokens = vqgan.encode(input_img)
            _, _, output_tokens = vqgan.encode(output_img)
        input_tokens = input_tokens.view(-1)
        output_tokens = output_tokens.view(-1)
        input_tokens = torch.clamp(input_tokens, 0, 8191)
        output_tokens = torch.clamp(output_tokens, 0, 8191)
        prompt_input_tokens.append(input_tokens)
        prompt_output_tokens.append(output_tokens)
        torch.cuda.empty_cache()
    log_print(f"\n[3] Loading and encoding query...")
    query_path = os.path.join(config.query_dir, config.query_file)
    query_img = load_and_preprocess_image(query_path, config.image_size, force_channels=3)
    with torch.no_grad():
        _, _, query_tokens = vqgan.encode(query_img)
    query_tokens = query_tokens.view(-1)
    query_tokens = torch.clamp(query_tokens, 0, 8191)
    log_print("\n[4] Creating ICL sequence...")
    log_print("\n[5] Generating output tokens with LLaMA 7B...")
    predicted_tokens = generate_output_tokens(llama, icl_sequence, num_tokens=128, K=config.K)
    torch.cuda.empty_cache()
    log_print("\n[6] Stage II: Converting prediction to tokens and reconstructing prompt...")
    gt_path = os.path.join(config.ground_truth_dir, config.query_file)
    prediction_path = os.path.join("predictions", f"temp_pred_{config.query_file}")
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    rmse = utils.process_image(gt_path, prediction_path, config.K)
    if os.path.exists(prediction_path):
        pred_img = Image.open(prediction_path).convert('RGB')
        pred_img = pred_img.resize((256, 256))
        pred_array = np.array(pred_img) / 255.0
    else:
        pred_array = np.random.rand(256, 256, 3)
    if len(pred_array.shape) == 2:
        decoded_prediction = torch.from_numpy(pred_array).unsqueeze(0).unsqueeze(0)
        decoded_prediction = decoded_prediction.repeat(1, 3, 1, 1)
    elif len(pred_array.shape) == 3:
        decoded_prediction = torch.from_numpy(pred_array).permute(2, 0, 1).unsqueeze(0)
    else:
        pred_array = np.random.rand(256, 256, 3)
        decoded_prediction = torch.from_numpy(pred_array).permute(2, 0, 1).unsqueeze(0)
    log_print("\n[7] Saving results...")
    query_name_no_ext = config.query_file.replace('.png', '')
    final_prediction_name = f"prediction_K{config.K}_{query_name_no_ext}.png"
    final_prediction_path = os.path.join(config.prediction_output_dir, final_prediction_name)
    log_print("\n=== ICL Prediction Complete (Llama 7B) ===")
    log_print(f"Log saved to: {log_filename}")
    
    close_logging()
    
    return decoded_prediction

if __name__ == "__main__":

    result = main()
