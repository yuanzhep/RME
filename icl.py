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
import argparse
import time
from datetime import datetime
from safetensors.torch import load_file
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def setup_logging(K: int, query_file: str) -> str:
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_name = os.path.splitext(query_file)[0]
    log_filename = f"log/ripple_K{K}_{query_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"RIPPLE Two-Stage - K={K}, Query={query_file}, Device={DEVICE}")
    logging.info("-" * 70)
    return log_filename

class RIPPLEConfig:
    def __init__(self, K: int = 3, query_file: Optional[str] = None):
        self.K = max(1, K)  
        self.query_file = query_file
        
        self.vqgan_config = ".../models/vqgan/config.json"
        self.vqgan_weights = ".../models/vqgan/pytorch_model.bin"
        self.llama_config = ".../models/llama/config.json"
        self.llama_model_dir = ".../models/llama"
        
        self.prompt_input_dir = ".../data/prompts/inputs"
        self.prompt_output_dir = ".../data/prompts/outputs"
        self.query_dir = ".../data/queries"
        self.prediction_output_dir = ".../predictions"
        self.database_dir = ".../database"
        
        os.makedirs(self.prediction_output_dir, exist_ok=True)
    
        self.image_size = 256 
        self.latent_size = 16  
        self.codebook_size = 8192  
        self.embedding_dim = 64  
        self.downsampling_factor = 16  
        self.max_sequence_length = 3500
        self.temperature = 0.8
        self.start_token = 0
        self.sep_token = 1  
        self.end_token = 2
        
        self.num_ripple_layers = 8  
        
        logging.info(f"Config initialized: K={self.K}, Image size={self.image_size}x{self.image_size}")

class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find nearest codebook entries
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # VQ loss
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, min_encoding_indices.view(z.shape[0], -1)
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.embedding(indices)
        if z_q.dim() == 3:
            z_q = z_q.unsqueeze(-1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

class VQGAN(nn.Module):
    def __init__(self, config_path: str, weights_path: str, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.config = self._load_config(config_path)
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.quantize = VectorQuantizer(
            n_embed=8192,
            embed_dim=self.embed_dim
        )
        
        if os.path.exists(weights_path):
            self._load_weights(weights_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load VQGAN config: {e}")
        
        return {
            "codebook_size": 8192,
            "embedding_dim": self.embed_dim,
            "n_embed": 8192,
        }
    
    def _load_weights(self, weights_path: str):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded VQGAN weights: {len(state_dict)} parameters")
        except Exception as e:
            logging.warning(f"Could not load VQGAN weights: {e}")
    
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
        h = self.encoder(x)  
        quant, emb_loss, indices = self.quantize(h)
        return h, emb_loss, indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            B, T = indices.shape
            H = W = int(math.sqrt(T))
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
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
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
        
        logging.info(f"LLaMA model initialized: {vocab_size} vocab, {hidden_size} hidden, {num_layers} layers")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load LLaMA configuration."""
        default_config = {
            "vocab_size": 8192,  
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
                for key in default_config:
                    if key in loaded_config:
                        default_config[key] = loaded_config[key]
            except Exception as e:
                logging.warning(f"Could not load LLaMA config: {e}")
        
        return default_config
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        input_ids = torch.clamp(input_ids, 0, self.config["vocab_size"] - 1)
        tok_emb = self.embedding(input_ids)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(pos_ids)
        
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class RIPPLESystem:
    def __init__(self, config: RIPPLEConfig):
        self.config = config
        
        logging.info("Loading VQGAN and LLaMA models...")
        self.vqgan = VQGAN(
            config.vqgan_config, 
            config.vqgan_weights, 
            config.embedding_dim
        ).to(DEVICE).eval()
        
        self.llama = SimpleLLaMA(
            config.llama_config, 
            config.llama_model_dir
        ).to(DEVICE).eval()
        
        logging.info("Models loaded successfully")
    
    def load_image(self, path: str) -> torch.Tensor:
        if not os.path.exists(path):
            logging.warning(f"Image not found: {path}, creating random tensor")
            return torch.randn(1, 3, self.config.image_size, self.config.image_size).to(DEVICE)
        
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.config.image_size, self.config.image_size), Image.Resampling.LANCZOS)
            tensor = T.ToTensor()(img).unsqueeze(0)
            return tensor.to(DEVICE)
        except Exception as e:
            logging.error(f"Error loading image {path}: {e}")
            return torch.randn(1, 3, self.config.image_size, self.config.image_size).to(DEVICE)
    
    def construct_prompts_case1(self, layout_dir: str) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
        """Case 1: Prompt construction within the same layout."""
        logging.info(f"Case 1: Constructing {self.config.K} prompts using spatial diversity")
        
        H, W = self.config.image_size, self.config.image_size
        positions = [(i, j) for i in range(20, H-20, 10) for j in range(20, W-20, 10)]
        
        if len(positions) < self.config.K:
            logging.warning(f"Not enough positions ({len(positions)}) for K={self.config.K}")
            selected_positions = positions
        else:
            positions_array = np.array(positions)
            kmeans = KMeans(n_clusters=self.config.K, random_state=42)
            clusters = kmeans.fit_predict(positions_array)
            
            # Select closest position to each centroid
            selected_positions = []
            for k in range(self.config.K):
                cluster_positions = positions_array[clusters == k]
                centroid = kmeans.cluster_centers_[k]
                distances = np.sum((cluster_positions - centroid) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                selected_positions.append(tuple(cluster_positions[closest_idx]))
        
        # Load prompts
        prompts = []
        for i, tx_pos in enumerate(selected_positions):
            input_path = os.path.join(layout_dir, f"input_{i}.png")
            output_path = os.path.join(layout_dir, f"output_{i}.png")
            
            input_img = self.load_image(input_path)
            output_img = self.load_image(output_path)
            
            prompts.append((input_img, output_img, tx_pos))
            logging.info(f"Loaded prompt {i+1}: Tx at {tx_pos}")
        
        return prompts
    
    def construct_prompts_case2(self, query_img: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
        """Case 2: Prompt retrieval from database using similarity."""
        logging.info(f"Case 2: Retrieving {self.config.K} prompts from database")
        
        with torch.no_grad():
            query_features = self.vqgan.encoder(query_img).flatten(1)
        
        database_path = os.path.join(self.config.database_dir, "database.json")
        if not os.path.exists(database_path):
            logging.warning("Database not found")
            return self._generate_random_prompts()
        
        try:
            with open(database_path, 'r') as f:
                database_info = json.load(f)
            
            similarities = []
            for entry in database_info:
                db_img = self.load_image(entry['input_path'])
                with torch.no_grad():
                    db_features = self.vqgan.encoder(db_img).flatten(1)
                
                sim = F.cosine_similarity(query_features, db_features, dim=1).item()
                similarities.append((sim, entry))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            selected_entries = similarities[:self.config.K]
            
            prompts = []
            for i, (sim, entry) in enumerate(selected_entries):
                input_img = self.load_image(entry['input_path'])
                output_img = self.load_image(entry['output_path'])
                tx_pos = tuple(entry['tx_position'])
                
                prompts.append((input_img, output_img, tx_pos))
                logging.info(f"Retrieved prompt {i+1}: similarity={sim:.3f}, Tx at {tx_pos}")
            
            return prompts
            
        except Exception as e:
            logging.error(f"Error loading database: {e}")
            return self._generate_random_prompts()
    
    def _generate_random_prompts(self) -> List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]]:
        prompts = []
        for i in range(self.config.K):
            input_img = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(DEVICE)
            output_img = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(DEVICE)
            tx_pos = (50 + i * 30, 50 + i * 30)
            prompts.append((input_img, output_img, tx_pos))
        return prompts
    
    def create_ripple_order(self, tokens: torch.Tensor, tx_pos: Tuple[int, int]) -> torch.Tensor:
        B, T = tokens.shape
        h = w = int(math.sqrt(T)) 
        
        tx_h = int(tx_pos[0] * h / self.config.image_size)
        tx_w = int(tx_pos[1] * w / self.config.image_size)
        
        coords = []
        for i in range(h):
            for j in range(w):
                dist = math.sqrt((i - tx_h) ** 2 + (j - tx_w) ** 2)
                coords.append((dist, i * w + j, tokens[0, i * w + j].item()))
        
        # ripple order
        coords.sort(key=lambda x: x[0])
        
        # ripple-ordered sequence
        ripple_tokens = torch.tensor([coord[2] for coord in coords], 
                                   dtype=tokens.dtype, device=tokens.device)
        
        return ripple_tokens.unsqueeze(0)
    
    def create_icl_sequence(self, prompts: List[Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]], 
                           query_tokens: torch.Tensor, query_tx_pos: Tuple[int, int]) -> torch.Tensor:
        sequence = []
        
        for input_img, output_img, tx_pos in prompts:
            with torch.no_grad():
                _, _, input_tokens = self.vqgan.encode(input_img)
                _, _, output_tokens = self.vqgan.encode(output_img)
            
            # ripple ordering
            input_ripple = self.create_ripple_order(input_tokens.view(1, -1), tx_pos)
            output_ripple = self.create_ripple_order(output_tokens.view(1, -1), tx_pos)
            
            # interleaved sequence
            for inp_token, out_token in zip(input_ripple[0], output_ripple[0]):
                sequence.extend([inp_token.item(), out_token.item()])
            
            sequence.append(self.config.sep_token) 
        
        # Add query with ripple ordering
        query_ripple = self.create_ripple_order(query_tokens.view(1, -1), query_tx_pos)
        sequence.extend(query_ripple[0].tolist())
        sequence.append(self.config.sep_token)
        
        return torch.tensor(sequence, dtype=torch.long, device=DEVICE)
    
    def stage1_generation(self, icl_sequence: torch.Tensor, num_tokens: int = 256) -> torch.Tensor:
        logging.info("Stage I: Causal autoregressive prediction")
        
        self.llama.eval()
        generated_tokens = []
        current_seq = icl_sequence.unsqueeze(0)
        
        with torch.no_grad():
            for step in range(num_tokens):
                logits = self.llama(current_seq)
                next_logits = logits[0, -1, :] / self.config.temperature
                next_logits[:10] = float('-inf')
                next_logits[self.config.codebook_size:] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_tokens.append(next_token.item())
                if next_token.item() == self.config.end_token:
                    break
        
        logging.info(f"Stage I generated {len(generated_tokens)} tokens")
        return torch.tensor(generated_tokens, dtype=torch.long, device=DEVICE)
    
    def stage2_refinement(self, icl_sequence: torch.Tensor, stage1_tokens: torch.Tensor, 
                         query_tokens: torch.Tensor, num_tokens: int = 256) -> torch.Tensor:
        logging.info("Stage II: Self-refinement")
        
        self_context = []
        query_flat = query_tokens.view(-1)
        
        # Interleave input and predicted tokens
        min_len = min(len(query_flat), len(stage1_tokens))
        for i in range(min_len):
            self_context.extend([query_flat[i].item(), stage1_tokens[i].item()])
        
        # refinement sequence P*_ref = [P*, P*_q]
        refinement_seq = torch.cat([
            icl_sequence,
            torch.tensor(self_context, dtype=torch.long, device=DEVICE),
            torch.tensor([self.config.sep_token], dtype=torch.long, device=DEVICE)
        ])
        
        self.llama.eval()
        generated_tokens = []
        current_seq = refinement_seq.unsqueeze(0)
        
        with torch.no_grad():
            for step in range(num_tokens):
                
                # predictions
                logits = self.llama(current_seq)
                next_logits = logits[0, -1, :] / self.config.temperature
                next_logits[:10] = float('-inf')
                next_logits[self.config.codebook_size:] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_tokens.append(next_token.item())
                
                if next_token.item() == self.config.end_token:
                    break
        
        logging.info(f"Stage II refined {len(generated_tokens)} tokens")
        return torch.tensor(generated_tokens, dtype=torch.long, device=DEVICE)
    
    def predict_radio_map(self, query_file: str, case: int = 1) -> Dict[str, Any]:
        """Complete two-stage radio map prediction pipeline."""
        logging.info(f"=== RIPPLE Two-Stage Prediction for {query_file} (Case {case}) ===")
        
        # Load query
        query_path = os.path.join(self.config.query_dir, query_file)
        query_img = self.load_image(query_path)
        query_tx_pos = (128, 128)  
        
        # Encode query
        with torch.no_grad():
            _, _, query_tokens = self.vqgan.encode(query_img)
        
        # Construct prompts based on case
        if case == 1:
            prompts = self.construct_prompts_case1(self.config.prompt_input_dir)
        else:
            prompts = self.construct_prompts_case2(query_img)
        
        # ripple ordering
        icl_sequence = self.create_icl_sequence(prompts, query_tokens, query_tx_pos)
        logging.info(f"ICL sequence length: {len(icl_sequence)}")
        
        # Stage I
        stage1_tokens = self.stage1_generation(icl_sequence, num_tokens=256)
        
        # Stage II
        stage2_tokens = self.stage2_refinement(icl_sequence, stage1_tokens, query_tokens, num_tokens=256)
        
        # Reconstruct final
        final_prediction = self.reconstruct_radio_map(stage2_tokens, query_tx_pos)
        results = self.save_results(query_file, final_prediction, stage1_tokens, stage2_tokens)
        
        logging.info("=== Two-Stage Prediction Complete ===")
        return results
    
    def reconstruct_radio_map(self, tokens: torch.Tensor, tx_pos: Tuple[int, int]) -> torch.Tensor:
        logging.info("Reconstructing radio map")
        
        try:
            num_tokens = len(tokens)
            grid_size = int(math.sqrt(num_tokens))
            if grid_size * grid_size > num_tokens:
                padding_size = grid_size * grid_size - num_tokens
                padding = torch.zeros(padding_size, dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, padding])
            elif grid_size * grid_size < num_tokens:
                tokens = tokens[:grid_size * grid_size]
            
            token_grid = tokens.view(1, grid_size, grid_size)
        
            # Decode using VQGAN decoder
            with torch.no_grad():
                reconstructed = self.vqgan.decode(token_grid)
            
            logging.info(f"Reconstructed radio map shape: {reconstructed.shape}")
            return reconstructed
            
        except Exception as e:
            logging.error(f"Error in reconstruction: {e}")
            # Return fallback random image
            return torch.randn(1, 3, self.config.image_size, self.config.image_size, device=DEVICE)
    
    def save_results(self, query_file: str, prediction: torch.Tensor, 
                    stage1_tokens: torch.Tensor, stage2_tokens: torch.Tensor) -> Dict[str, Any]:
        """Save prediction results and return metrics."""
        query_name = os.path.splitext(query_file)[0]
        
        # Save prediction
        prediction_filename = f"ripple_K{self.config.K}_{query_name}_two_stage.png"
        prediction_path = os.path.join(self.config.prediction_output_dir, prediction_filename)
        img_tensor = prediction.squeeze(0).cpu()
        img_tensor = torch.clamp((img_tensor + 1.0) / 2.0, 0.0, 1.0)  # Normalize [-1,1] to [0,1]
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img.save(prediction_path)
        logging.info(f"Saved prediction to: {prediction_path}")
        
        stage1_filename = f"stage1_K{self.config.K}_{query_name}.png"
        stage1_path = os.path.join(self.config.prediction_output_dir, stage1_filename)
        
        try:
            stage1_prediction = self.reconstruct_radio_map(stage1_tokens, (128, 128))
            stage1_img_tensor = stage1_prediction.squeeze(0).cpu()
            stage1_img_tensor = torch.clamp((stage1_img_tensor + 1.0) / 2.0, 0.0, 1.0)
            stage1_img_array = stage1_img_tensor.permute(1, 2, 0).numpy()
            stage1_img_array = (stage1_img_array * 255).astype(np.uint8)
            stage1_img = Image.fromarray(stage1_img_array)
            stage1_img.save(stage1_path)
            logging.info(f"Saved Stage I result to: {stage1_path}")
        except Exception as e:
            logging.warning(f"Could not save Stage I result: {e}")
        
        gt_path = os.path.join(self.config.ground_truth_dir, query_file)
        metrics = self.compute_metrics(prediction_path, gt_path)
        
        tokens_data = {
            'stage1_tokens': stage1_tokens.cpu().tolist(),
            'stage2_tokens': stage2_tokens.cpu().tolist(),
            'metrics': metrics,
            'config': {
                'K': self.config.K,
                'temperature': self.config.temperature,
                'image_size': self.config.image_size,
                'codebook_size': self.config.codebook_size
            }
        }
        
        tokens_filename = f"tokens_K{self.config.K}_{query_name}.json"
        tokens_path = os.path.join(self.config.prediction_output_dir, tokens_filename)
        
        with open(tokens_path, 'w') as f:
            json.dump(tokens_data, f, indent=2)
        
        return {
            'prediction_path': prediction_path,
            'stage1_path': stage1_path,
            'tokens_path': tokens_path,
            'metrics': metrics,
            'stage1_tokens_count': len(stage1_tokens),
            'stage2_tokens_count': len(stage2_tokens)
        }
    
    def compute_metrics(self, pred_path: str, gt_path: str) -> Dict[str, float]:
        """Compute evaluation metrics between prediction and ground truth."""
        try:
            if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                return {"rmse": float('inf'), "mae": float('inf'), "psnr": 0.0}
            
            pred_img = Image.open(pred_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            pred_img = pred_img.resize((self.config.image_size, self.config.image_size))
            gt_img = gt_img.resize((self.config.image_size, self.config.image_size))
            
            pred_array = np.array(pred_img, dtype=np.float32) / 255.0
            gt_array = np.array(gt_img, dtype=np.float32) / 255.0
            
            mse = np.mean((pred_array - gt_array) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_array - gt_array))
            
            if mse > 0:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            else:
                psnr = float('inf')
            
            logging.info(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, PSNR: {psnr:.2f} dB")
            return {"rmse": rmse, "mae": mae, "psnr": psnr}
            
        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
            return {"rmse": float('inf'), "mae": float('inf'), "psnr": 0.0}
    
    def analyze_ripple_structure(self, tokens: torch.Tensor, tx_pos: Tuple[int, int]) -> Dict[str, Any]:
        logging.info("Analyzing ripple structure...")
        
        try:
            num_tokens = len(tokens)
            grid_size = int(math.sqrt(num_tokens))
            
            if grid_size * grid_size != num_tokens:
                logging.warning(f"Token count {num_tokens} is not a perfect square")
                return {}
            
            token_grid = tokens.view(grid_size, grid_size).cpu().numpy()
            
            tx_h = int(tx_pos[0] * grid_size / self.config.image_size)
            tx_w = int(tx_pos[1] * grid_size / self.config.image_size)
            
            distances = []
            token_values = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    dist = math.sqrt((i - tx_h) ** 2 + (j - tx_w) ** 2)
                    distances.append(dist)
                    token_values.append(token_grid[i, j])
            
            distances = np.array(distances)
            token_values = np.array(token_values)
            
            correlation = np.corrcoef(distances, token_values)[0, 1]
            
            sorted_indices = np.argsort(distances)
            ripple_order_consistency = 0
            for i in range(len(sorted_indices) - 1):
                if token_values[sorted_indices[i]] <= token_values[sorted_indices[i + 1]]:
                    ripple_order_consistency += 1
            
            ripple_order_consistency /= (len(sorted_indices) - 1)
            
            return {
                'distance_token_correlation': float(correlation),
                'ripple_order_consistency': float(ripple_order_consistency),
                'mean_distance': float(np.mean(distances)),
                'max_distance': float(np.max(distances)),
                'token_range': [int(np.min(token_values)), int(np.max(token_values))]
            }
            
        except Exception as e:
            logging.error(f"Error analyzing ripple structure: {e}")
            return {}

def create_visualization(results: Dict[str, Any], config: RIPPLEConfig):
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if os.path.exists(results['stage1_path']):
            stage1_img = Image.open(results['stage1_path'])
            axes[0].imshow(stage1_img)
            axes[0].set_title('Stage I: Causal Prediction')
            axes[0].axis('off')
        
        if os.path.exists(results['prediction_path']):
            stage2_img = Image.open(results['prediction_path'])
            axes[1].imshow(stage2_img)
            axes[1].set_title('Stage II: Refined Prediction')
            axes[1].axis('off')
        
        metrics = results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[2].bar(metric_names, metric_values)
        axes[2].set_title('Evaluation Metrics')
        axes[2].set_ylabel('Value')
        
        plt.tight_layout()
        
        viz_path = os.path.join(config.prediction_output_dir, 'two_stage_comparison.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualization saved to: {viz_path}")
        
    except Exception as e:
        logging.warning(f"Could not create visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description='RIPPLE Two-Stage')
    parser.add_argument('--K', type=int, default=3, help='Number of prompts (default: 3)')
    parser.add_argument('--query', type=str, default="B24_Ant1_f1_S0.png", help='Query file name')
    parser.add_argument('--case', type=int, choices=[1, 2], default=1, 
                       help='Case 1: same layout, Case 2: different layouts (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--analyze', action='store_true', help='Perform ripple structure analysis')
    parser.add_argument('--visualize', action='store_true', help='Create result visualizations')
    
    args = parser.parse_args()

    log_filename = setup_logging(args.K, args.query)
    
    try:
        config = RIPPLEConfig(K=args.K, query_file=args.query)
        config.temperature = args.temperature
        ripple_system = RIPPLESystem(config)

        results = ripple_system.predict_radio_map(args.query, case=args.case)
        
        if args.analyze:
            logging.info("Performing ripple structure analysis...")
            
            with open(results['tokens_path'], 'r') as f:
                tokens_data = json.load(f)
            
            stage1_tokens = torch.tensor(tokens_data['stage1_tokens'])
            stage2_tokens = torch.tensor(tokens_data['stage2_tokens'])
            
            stage1_analysis = ripple_system.analyze_ripple_structure(stage1_tokens, (128, 128))
            stage2_analysis = ripple_system.analyze_ripple_structure(stage2_tokens, (128, 128))
            
            logging.info("Stage I Analysis:")
            for key, value in stage1_analysis.items():
                logging.info(f"  {key}: {value}")
            
            logging.info("Stage II Analysis:")
            for key, value in stage2_analysis.items():
                logging.info(f"  {key}: {value}")
        
        if args.visualize:
            create_visualization(results, config)
        
        logging.info("\n" + "="*70)
        logging.info("RIPPLE TWO-STAGE PREDICTION SUMMARY")
        logging.info("="*70)
        logging.info(f"Query: {args.query}")
        logging.info(f"Case: {args.case}")
        logging.info(f"K (prompts): {args.K}")
        logging.info(f"Temperature: {args.temperature}")
        logging.info(f"Stage I tokens: {results['stage1_tokens_count']}")
        logging.info(f"Stage II tokens: {results['stage2_tokens_count']}")
        logging.info(f"Final prediction: {results['prediction_path']}")
        
        metrics = results['metrics']
        logging.info(f"RMSE: {metrics['rmse']:.4f}")
        logging.info(f"MAE: {metrics['mae']:.4f}")
        logging.info(f"PSNR: {metrics['psnr']:.2f} dB")
        logging.info("="*70)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    results = main()
    if results is None:
        exit(1)