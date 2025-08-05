import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import time
import sys
from datetime import datetime
from safetensors.torch import load_file
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def setup_logging(K: int, query_file: str) -> str:
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = os.path.splitext(query_file)[0]  # More robust file extension removal
    log_filename = f"log/ripple_K{K}_{query_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Log started - Parameters: K={K}, Query={query_file}, Device: {DEVICE}")
    logging.info("-" * 50)
    return log_filename

def load_safetensors_sharded(model_dir: str) -> Dict[str, torch.Tensor]:
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    try:
        with open(index_file, 'r') as f:
            index = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in index file: {e}")
    
    if "weight_map" not in index:
        raise KeyError("Index file missing 'weight_map' key")
    
    state_dict = {}
    weight_map = index["weight_map"]
    shard_files = list(set(weight_map.values()))
    
    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        if os.path.exists(shard_path):
            logging.info(f"Loading shard: {shard_file}")
            try:
                shard_weights = load_file(shard_path)
                state_dict.update(shard_weights)
            except Exception as e:
                logging.error(f"Failed to load shard {shard_file}: {e}")
                raise
        else:
            logging.warning(f"Shard file not found: {shard_path}")
    
    return state_dict

class Config:
    def __init__(self, K: int = 2, query_file: Optional[str] = None):
        self.K = max(0, K)  
        self.query_file = query_file
        self.vqgan_config = self._validate_path(".../models/config.json", "VQGAN config")
        self.vqgan_weights = self._validate_path(".../models/pytorch_model.bin", "VQGAN weights")
        self.llama_config = self._validate_path(".../models/LVM_ckpts/config.json", "LLaMA config")
        self.llama_model_dir = self._validate_path(".../models/LVM_ckpts", "LLaMA model directory")
        self.prompt_input_dir = ".../prompts/Inputs"
        self.prompt_output_dir = ".../prompts/Outputs"
        self.query_dir = ".../queries"
        self.prediction_output_dir = ".../predictions"
        
        os.makedirs(self.prediction_output_dir, exist_ok=True)
        self.image_size = 256
        self.latent_size = 16
        self.codebook_size = 8192
        self.temperature = 0.8
        self.max_sequence_length = 3500
        # self.start_token = 0
        self.sep_token = 1
        self.end_token = 2
        self.available_prompts = self._get_available_prompts()
        self.selected_prompts = self.available_prompts[:self.K] if self.K > 0 else []
        
        logging.info(f"Config initialized with K={self.K}, selected {len(self.selected_prompts)} prompts")
    
    def _validate_path(self, path: str, description: str) -> str:
        if not os.path.exists(path):
            logging.warning(f"{description} path does not exist: {path}")
        return path
    
    def _get_available_prompts(self) -> List[str]:
        if not os.path.exists(self.prompt_input_dir):
            logging.warning(f"Prompt input directory not found: {self.prompt_input_dir}")
            return []
        
        try:
            files = os.listdir(self.prompt_input_dir)
            prompt_names = [os.path.splitext(f)[0] for f in files if f.endswith('.png')]
            logging.info(f"Found {len(prompt_names)} available prompts")
            return sorted(prompt_names)
        except OSError as e:
            logging.error(f"Error reading prompt directory: {e}")
            return []

class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {z.dim()}D")
        
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
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D indices tensor, got {indices.dim()}D")
        
        if indices.dim() == 3:
            B, H, W = indices.shape
            indices = indices.view(B, H, W)
        
        z_q = self.embedding(indices)
        if z_q.dim() == 3:  
            z_q = z_q.unsqueeze(-1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class VQGAN(nn.Module):
    """VQGAN model with proper initialization."""
    
    def __init__(self, config_path: str, weights_path: str):
        super().__init__()
        self.embed_dim = 256
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'quantize.embedding.weight' in state_dict:
                    self.embed_dim = state_dict['quantize.embedding.weight'].shape[1]
                    logging.info(f"Detected embedding dimension: {self.embed_dim}")
            except Exception as e:
                logging.warning(f"Could not detect embedding dimension: {e}")
        
        self.config = self._load_config(config_path)
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.quantize = VectorQuantizer(
            n_embed=self.config.get("n_embed", 8192),
            embed_dim=self.embed_dim
        )
        
        if os.path.exists(weights_path):
            self._load_weights(weights_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logging.info("Loaded VQGAN config")
                return config
            except Exception as e:
                logging.warning(f"Could not load config: {e}")
        
        return {
            "codebook_size": 8192,
            "embedding_dim": self.embed_dim,
            "n_embed": 8192,
        }
    
    def _load_weights(self, weights_path: str):
        """Load model weights with error handling."""
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys in state dict: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in state dict: {unexpected_keys}")
            
            logging.info(f"Loaded VQGAN weights from {weights_path}")
        except Exception as e:
            logging.error(f"Could not load weights: {e}")
    
    def _make_encoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.embed_dim, 4, 2, 1),
        )
    
    def _make_decoder(self) -> nn.Module:
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected 4D tensor with 3 channels, got shape {x.shape}")
        
        h = self.encoder(x)
        quant, emb_loss, indices = self.quantize(h)
        return h, emb_loss, indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            B, T = indices.shape
            H = W = int(math.sqrt(T))
            if H * W != T:
                raise ValueError(f"Cannot reshape {T} tokens to square grid")
            indices = indices.view(B, H, W)
        
        quant = self.quantize.get_codebook_entry(indices)
        dec = self.decoder(quant)
        return dec

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False)
        )
        self.ln1 = RMSNorm(hidden_size)
        self.ln2 = RMSNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        normed_x = self.ln1(x)
        attn_out, _ = self.attention(normed_x, normed_x, normed_x, attn_mask=causal_mask)
        x = x + attn_out
        normed_x = self.ln2(x)
        ff_out = self.feed_forward(normed_x)
        x = x + ff_out
        
        return x

class SimpleLLaMA(nn.Module):
    def __init__(self, config_path: str, model_dir: str):
        super().__init__()
        self.config = self._load_config(config_path)
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
        
        self._load_weights(model_dir)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        default_config = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "max_seq_len": 4096,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-6,
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                config = {
                    "vocab_size": loaded_config.get("vocab_size", default_config["vocab_size"]),
                    "hidden_size": loaded_config.get("hidden_size", default_config["hidden_size"]),
                    "num_layers": loaded_config.get("num_hidden_layers", default_config["num_layers"]),
                    "num_heads": loaded_config.get("num_attention_heads", default_config["num_heads"]),
                    "max_seq_len": loaded_config.get("max_position_embeddings", default_config["max_seq_len"]),
                    "intermediate_size": loaded_config.get("intermediate_size", default_config["intermediate_size"]),
                    "rms_norm_eps": loaded_config.get("rms_norm_eps", default_config["rms_norm_eps"]),
                }
                
                logging.info(f"Loaded LLaMA config: {config}")
                return config
                
            except Exception as e:
                logging.warning(f"Could not load config, using defaults: {e}")
        
        return default_config
    
    def _load_weights(self, model_dir: str):
        try:
            logging.info("Loading LLaMA weights from safetensors...")
            state_dict = load_safetensors_sharded(model_dir)
            self._map_llama_weights(state_dict)
            logging.info("Successfully loaded LLaMA weights")
        except Exception as e:
            logging.warning(f"Could not load LLaMA weights: {e}")
    
    def _map_llama_weights(self, state_dict: Dict[str, torch.Tensor]):
        model_dict = self.state_dict()
        weight_mapping = {}
        if "model.embed_tokens.weight" in state_dict:
            weight_mapping["embedding.weight"] = "model.embed_tokens.weight"
        
        # Transformer layers
        for i in range(self.config["num_layers"]):
            llama_prefix = f"model.layers.{i}"
            model_prefix = f"layers.{i}"
            
            # Feed forward layers
            if f"{llama_prefix}.mlp.gate_proj.weight" in state_dict:
                weight_mapping[f"{model_prefix}.feed_forward.0.weight"] = f"{llama_prefix}.mlp.gate_proj.weight"
            if f"{llama_prefix}.mlp.down_proj.weight" in state_dict:
                weight_mapping[f"{model_prefix}.feed_forward.2.weight"] = f"{llama_prefix}.mlp.down_proj.weight"
            
            # Layer norms
            if f"{llama_prefix}.input_layernorm.weight" in state_dict:
                weight_mapping[f"{model_prefix}.ln1.weight"] = f"{llama_prefix}.input_layernorm.weight"
            if f"{llama_prefix}.post_attention_layernorm.weight" in state_dict:
                weight_mapping[f"{model_prefix}.ln2.weight"] = f"{llama_prefix}.post_attention_layernorm.weight"
        
        # layer norm and head
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
                    logging.warning(f"Shape mismatch for {model_key}: {llama_weight.shape} vs {model_weight_shape}")
        
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)
        logging.info(f"Loaded {len(filtered_dict)} compatible weight tensors")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"Expected 2D input_ids tensor, got {input_ids.dim()}D")
        
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

def load_and_preprocess_image(path: str, target_size: int = 256, force_channels: Optional[int] = None) -> torch.Tensor:
    if not os.path.exists(path):
        logging.warning(f"File {path} not found, creating random tensor")
        channels = force_channels if force_channels else 3
        return torch.randn(1, channels, target_size, target_size).to(DEVICE)
    
    try:
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
    
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        channels = force_channels if force_channels else 3
        return torch.randn(1, channels, target_size, target_size).to(DEVICE)

def create_icl_sequence(input_tokens_list: List[torch.Tensor], 
                       output_tokens_list: List[torch.Tensor],
                       query_tokens: torch.Tensor,
                       max_length: int = 3500,
                       vocab_size: int = 8192) -> torch.Tensor:
    if len(input_tokens_list) != len(output_tokens_list):
        raise ValueError("Input and output token lists must have same length")
    
    sequence = []
    
    for inp_tokens, out_tokens in zip(input_tokens_list, output_tokens_list):
        inp_tokens_truncated = inp_tokens[:tokens_per_prompt]
        out_tokens_truncated = out_tokens[:tokens_per_prompt]
        
        inp_tokens_truncated = torch.clamp(inp_tokens_truncated, 10, vocab_size - 10)
        out_tokens_truncated = torch.clamp(out_tokens_truncated, 10, vocab_size - 10)
        
        sequence.extend(inp_tokens_truncated.tolist())
        sequence.append(1)  # separator token
        sequence.extend(out_tokens_truncated.tolist())
        sequence.append(2)  # end token
    
    # Add query
    query_tokens_truncated = query_tokens[:tokens_per_prompt]
    query_tokens_truncated = torch.clamp(query_tokens_truncated, 10, vocab_size - 10)
    
    sequence.extend(query_tokens_truncated.tolist())
    sequence.append(1)  # separator token
    sequence = [max(0, min(token, vocab_size - 1)) for token in sequence]
    
    return torch.tensor(sequence, dtype=torch.long, device=DEVICE)

def generate_output_tokens(model: SimpleLLaMA, input_sequence: torch.Tensor, 
                         num_tokens: int = 128, temperature: float = 0.8) -> torch.Tensor:
    model.eval()
    generated_tokens = []
    vocab_size = model.config["vocab_size"]
    current_seq = input_sequence.unsqueeze(0)
    current_seq = torch.clamp(current_seq, 0, vocab_size - 1)
    num_tokens = min(num_tokens, 256)  
    
    with torch.no_grad():
        for step in range(num_tokens):
            try:
                context_limit = 4096
                logits = model(current_seq)
                next_logits = logits[0, -1, :]
                next_logits = next_logits / temperature
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
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_tokens.append(next_token_val)
                
                if next_token_val == 2:  
                    logging.info(f"Generated {len(generated_tokens)} tokens (stopped at end token)")
                    break
                    
            except Exception as e:
                logging.error(f"Error during generation at step {step}: {e}")
                break
    
    return torch.tensor(generated_tokens, dtype=torch.long, device=DEVICE)

def save_generated_image(tokens: torch.Tensor, vqgan: VQGAN, output_path: str, image_size: int = 256):
    try:
        if tokens.dim() == 1:
            grid_size = int(math.sqrt(len(tokens)))
            if grid_size * grid_size != len(tokens):
                target_tokens = grid_size * grid_size
                if len(tokens) < target_tokens:
                    padding = torch.zeros(target_tokens - len(tokens), dtype=tokens.dtype, device=tokens.device)
                    tokens = torch.cat([tokens, padding])
                else:
                    tokens = tokens[:target_tokens]
            
            tokens = tokens.view(1, grid_size, grid_size)
        
        with torch.no_grad():
            decoded_img = vqgan.decode(tokens)
            
        img_tensor = decoded_img.squeeze(0).cpu()
        img_tensor = torch.clamp((img_tensor + 1.0) / 2.0, 0.0, 1.0) 
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        logging.info(f"Saved generated image to {output_path}")
        
        return img
        
    except Exception as e:
        logging.error(f"Error saving generated image: {e}")
        fallback_img = Image.fromarray(np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8))
        fallback_img.save(output_path)
        return fallback_img

def compute_metrics(predicted_path: str, ground_truth_path: str) -> Dict[str, float]:
    try:
        if not os.path.exists(predicted_path):
            logging.warning(f"Predicted image not found: {predicted_path}")
            return {"rmse": float('inf'), "mae": float('inf'), "psnr": 0.0}
        
        if not os.path.exists(ground_truth_path):
            logging.warning(f"Ground truth image not found: {ground_truth_path}")
            return {"rmse": float('inf'), "mae": float('inf'), "psnr": 0.0}
        
        pred_img = Image.open(predicted_path).convert('RGB')
        gt_img = Image.open(ground_truth_path).convert('RGB')
        
        pred_img = pred_img.resize((256, 256))
        gt_img = gt_img.resize((256, 256))
        
        pred_array = np.array(pred_img, dtype=np.float32) / 255.0
        gt_array = np.array(gt_img, dtype=np.float32) / 255.0
        
        mse = np.mean((pred_array - gt_array) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_array - gt_array))
        
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        return {"rmse": rmse, "mae": mae, "psnr": psnr}
        
    except Exception as e:
        logging.error(f"Error computing metrics: {e}")
        return {"rmse": float('inf'), "mae": float('inf'), "psnr": 0.0}

def main():
    parser = argparse.ArgumentParser(description='RIPPLE')
    parser.add_argument('--K', type=int, default=3, help='Number of prompts to use (default: 3)')
    parser.add_argument('--query', type=str, default="B24_Ant1_f1_S0.png", help='Query file name')
    parser.add_argument('--num_tokens', type=int, default=256, help='Number of tokens (default: 256)')
    args = parser.parse_args()
    log_filename = setup_logging(args.K, args.query)
    try:
        config = Config(K=args.K, query_file=args.query)
        logging.info("=== RIPPLE (LLaMA 7B) ICL System ===")
        if config.K > 0 and len(config.selected_prompts) == 0:
            logging.error("No prompt examples found!")
            return None
        
        logging.info(f"[1] Loading models...")
        
        # Load VQGAN
        vqgan = VQGAN(config.vqgan_config, config.vqgan_weights).to(DEVICE).eval()
        logging.info("VQGAN loaded")
        
        # Load LLaMA
        llama = SimpleLLaMA(config.llama_config, config.llama_model_dir).to(DEVICE).eval()
        logging.info("LLaMA loaded")
        
        logging.info(f"[2] Loading prompt examples (K={config.K})...")
        
        prompt_input_tokens = []
        prompt_output_tokens = []
        
        for i, name in enumerate(config.selected_prompts):
            logging.info(f"Processing prompt {i+1}/{len(config.selected_prompts)}: {name}")
            
            input_path = os.path.join(config.prompt_input_dir, name + ".png")
            output_path = os.path.join(config.prompt_output_dir, name + ".png")
            
            input_img = load_and_preprocess_image(input_path, config.image_size, force_channels=3)
            output_img = load_and_preprocess_image(output_path, config.image_size, force_channels=3)
            
            with torch.no_grad():
                _, _, input_tokens = vqgan.encode(input_img)
                _, _, output_tokens = vqgan.encode(output_img)
            
            # Flatten
            input_tokens = input_tokens.view(-1)
            output_tokens = output_tokens.view(-1)
            prompt_input_tokens.append(input_tokens)
            prompt_output_tokens.append(output_tokens)

        logging.info(f"[3] Loading and encoding query: {config.query_file}")
        
        # Load query
        query_path = os.path.join(config.query_dir, config.query_file)
        query_img = load_and_preprocess_image(query_path, config.image_size, force_channels=3)
        
        with torch.no_grad():
            _, _, query_tokens = vqgan.encode(query_img)
        
        query_tokens = query_tokens.view(-1)
        query_tokens = torch.clamp(query_tokens, 0, config.codebook_size - 1)
        
        logging.info(f"[4] Creating sequence...")
        
        icl_sequence = create_icl_sequence(
            prompt_input_tokens, 
            prompt_output_tokens, 
            query_tokens,
            max_length=config.max_sequence_length,
            vocab_size=config.codebook_size
        )
        
        logging.info(f"ICL sequence length: {len(icl_sequence)}")
        logging.info(f"[5] Generating output tokens with LLaMA...")
        
        predicted_tokens = generate_output_tokens(
            llama, 
            icl_sequence, 
            num_tokens=args.num_tokens,
            temperature=args.temperature
        )
        
        logging.info(f"Generated {len(predicted_tokens)} tokens")
        logging.info(f"[6] Reconstructing image from tokens...")
        query_name_no_ext = os.path.splitext(config.query_file)[0]
        prediction_filename = f"prediction_K{config.K}_{query_name_no_ext}.png"
        prediction_path = os.path.join(config.prediction_output_dir, prediction_filename)
        generated_img = save_generated_image(predicted_tokens, vqgan, prediction_path, config.image_size)
        logging.info(f"[7] Computing evaluation metrics...")
        gt_path = os.path.join(config.ground_truth_dir, config.query_file)
        metrics = compute_metrics(prediction_path, gt_path)
        logging.info(f"Evaluation metrics:")
        logging.info(f"  RMSE: {metrics['rmse']:.3f}")
        logging.info(f"[8] Saving results...")
        logging.info(f"Prediction saved to: {prediction_path}")
        logging.info("=== ICL Complete ===")
        logging.info(f"Log saved to: {log_filename}")
        
        return {
            'prediction_path': prediction_path,
            'metrics': metrics,
            'generated_tokens': predicted_tokens,
            'config': config
        }
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)