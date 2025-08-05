import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RadioMapDataset(Dataset):
    def __init__(self, scene_dir, pathloss_dir, image_size=256, split='train', train_ratio=0.8):
        self.scene_dir = Path(scene_dir)
        self.pathloss_dir = Path(pathloss_dir)
        self.image_size = image_size
        
        scene_files = list(self.scene_dir.glob("*.png"))
        self.file_pairs = []
        
        for scene_file in scene_files:
            pathloss_file = self.pathloss_dir / scene_file.name
            if pathloss_file.exists():
                self.file_pairs.append((scene_file, pathloss_file))
        
        # Train/val split
        n_files = len(self.file_pairs)
        n_train = int(n_files * train_ratio)
        
        if split == 'train':
            self.file_pairs = self.file_pairs[:n_train]
        else:
            self.file_pairs = self.file_pairs[n_train:]
        
        print(f"{split} dataset: {len(self.file_pairs)} pairs")
        
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),  # Small rotations
                T.ColorJitter(brightness=0.1),  # Brightness perturbations
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor()
            ])
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        scene_path, pathloss_path = self.file_pairs[idx]
        scene_img = Image.open(scene_path).convert('RGB')
        scene_tensor = self.transform(scene_img)  # [3, H, W]
        pathloss_img = Image.open(pathloss_path).convert('L')
        pathloss_tensor = self.transform(pathloss_img)  # [1, H, W]
        pathloss_tensor = pathloss_tensor.repeat(3, 1, 1)  # [3, H, W]
        
        return {
            'scene': scene_tensor,
            'pathloss': pathloss_tensor,
            'filename': scene_path.stem
        }

class FrozenVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, pretrained_codebook=None, beta=0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(n_embed, embed_dim)
        
        if pretrained_codebook is not None:
            print(f"Loading frozen codebook: {pretrained_codebook.shape}")
            self.embedding.weight.data.copy_(pretrained_codebook)
            self.embedding.weight.requires_grad = False  # FREEZE CODEBOOK
            print("Codebook frozen - token space preserved!")
        else:
            self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
            print("Warning: No pretrained codebook provided")
    
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        z_flattened = z.view(-1, self.embed_dim)  # [B*H*W, C]
        
        # Find nearest codebook entries (frozen)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Commitment loss: ||ẑ - sg(z)||_2^2
        # Only encoder (ẑ) gets gradients, codebook entry (z) is stopped
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        
        # Straight-through estimator for gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return z_q, commitment_loss, min_encoding_indices.view(z.shape[0], -1)
    
    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)  # [B, H, W, C]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return z_q

class ResnetBlock(nn.Module):
    """Residual block for encoder/decoder"""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        
        return x + h

class Encoder(nn.Module):
    """Encoder for radio scene tensors (distance, reflectance, transmittance)"""
    def __init__(self, ch=128, ch_mult=(1,2,4,8), num_res_blocks=2, 
                 in_channels=3, resolution=256, z_channels=256):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        # Input convolution for 3-channel scene tensors
        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, 1, 1)
        
        # Downsampling
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res <= 32:
                    attn.append(nn.GroupNorm(32, block_in))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = nn.Conv2d(block_in, block_in, 3, 2, 1)
                curr_res = curr_res // 2
            self.down.append(down)
        
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = nn.GroupNorm(32, block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        
        # Output
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, 3, 1, 1)
    
    def forward(self, x):
        hs = [self.conv_in(x)]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    """Decoder for pathloss map reconstruction"""
    
    def __init__(self, ch=128, out_ch=3, ch_mult=(1,2,4,8), num_res_blocks=2,
                 resolution=256, z_channels=256):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        
        # Input from quantized features
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)
        
        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = nn.GroupNorm(32, block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        
        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if curr_res <= 32:
                    attn.append(nn.GroupNorm(32, block_in))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = nn.ConvTranspose2d(block_in, block_in, 4, 2, 1)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        
        # Output for pathloss maps
        self.norm_out = nn.GroupNorm(32, block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)
    
    def forward(self, z):
        h = self.conv_in(z)
        
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        
        sequence = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.GroupNorm(32, ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            nn.GroupNorm(32, ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)

class FrozenCodebookVQGAN(nn.Module):
    """VQGAN with frozen codebook for radio map domain"""
    def __init__(self, embed_dim=256, n_embed=8192, ch=128, resolution=256,
                 pretrained_codebook=None):
        super().__init__()
        
        self.encoder = Encoder(
            ch=ch, in_channels=3, resolution=resolution, z_channels=embed_dim
        )
        self.decoder = Decoder(
            ch=ch, out_ch=3, resolution=resolution, z_channels=embed_dim
        )
        self.quantize = FrozenVectorQuantizer(
            n_embed=n_embed, embed_dim=embed_dim, pretrained_codebook=pretrained_codebook
        )
        print(f"Frozen Codebook VQGAN:")
        print(f"   Codebook: {n_embed} entries, {embed_dim}D")
        print(f"   Frozen: {not self.quantize.embedding.weight.requires_grad}")
    
    def encode(self, x):
        h = self.encoder(x)
        quant, commitment_loss, indices = self.quantize(h)
        return quant, commitment_loss, indices
    
    def decode(self, quant):
        return self.decoder(quant)
    
    def forward(self, input_data, mode='scene'):
        """
        mode: 'scene' (train encoder) or 'pathloss' (train decoder)
        """
        if mode == 'scene':
            # Scene tensor processing: train encoder, freeze decoder
            quant, commitment_loss, indices = self.encode(input_data)
            with torch.no_grad():
                reconstruction = self.decode(quant)
        elif mode == 'pathloss':
            # Pathloss processing: train decoder, freeze encoder
            with torch.no_grad():
                h = self.encoder(input_data)
                quant, commitment_loss, indices = self.quantize(h)
            reconstruction = self.decode(quant)
        else:  # validation mode
            quant, commitment_loss, indices = self.encode(input_data)
            reconstruction = self.decode(quant)
        
        return reconstruction, commitment_loss, indices

def load_pretrained_codebook(checkpoint_path):
    """Extract codebook from existing VQGAN checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        if checkpoint_path.endswith('.pth'):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            possible_keys = [
                'quantize.embedding.weight',
                'model_state_dict.quantize.embedding.weight',
                'vqgan_state_dict.quantize.embedding.weight'
            ]
            
            for key in possible_keys:
                if key in checkpoint:
                    codebook = checkpoint[key]
                    print(f"Found codebook at key: {key}")
                    print(f"   Shape: {codebook.shape}")
                    return codebook
                elif 'model_state_dict' in checkpoint and key.replace('model_state_dict.', '') in checkpoint['model_state_dict']:
                    codebook = checkpoint['model_state_dict'][key.replace('model_state_dict.', '')]
                    print(f"Found codebook in model_state_dict")
                    print(f"   Shape: {codebook.shape}")
                    return codebook
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'quantize.embedding.weight' in state_dict:
                codebook = state_dict['quantize.embedding.weight']
                print(f"Found codebook in direct state dict")
                print(f"   Shape: {codebook.shape}")
                return codebook
        
        print("No codebook found")
        return None
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def train_frozen_codebook_vqgan(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Load frozen codebook
    pretrained_codebook = load_pretrained_codebook(args.pretrained_codebook_path)
    if pretrained_codebook is None:
        raise ValueError("Could not load pretrained codebook")
    
    # Datasets
    train_dataset = RadioMapDataset(
        args.scene_dir, args.pathloss_dir, 
        image_size=args.image_size, split='train'
    )
    val_dataset = RadioMapDataset(
        args.scene_dir, args.pathloss_dir, 
        image_size=args.image_size, split='val'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=args.num_workers
    )
    
    # Models
    vqgan = FrozenCodebookVQGAN(
        embed_dim=args.embed_dim,
        n_embed=args.n_embed,
        ch=args.ch,
        resolution=args.image_size,
        pretrained_codebook=pretrained_codebook
    ).to(DEVICE)
    
    # Discriminator G(·) as per paper notation
    discriminator_G = Discriminator(input_nc=3).to(DEVICE)
    
    # Optimizers as specified: Adam, lr=1e-4
    opt_encoder = optim.Adam(vqgan.encoder.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_decoder = optim.Adam(vqgan.decoder.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_disc = optim.Adam(discriminator_G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    
    # Loss functions
    l2_loss = nn.MSELoss()  # L_rec = L2 loss as specified
    bce_loss = nn.BCEWithLogitsLoss()  # For adversarial loss
    
    print("Starting Frozen Codebook Training")
    print(f"Parameters: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Commitment weight λ = {args.commitment_weight}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        vqgan.train()
        discriminator_G.train()
        
        epoch_stats = {
            'scene_batches': 0, 'pathloss_batches': 0,
            'encoder_loss': 0, 'decoder_loss': 0, 'disc_loss': 0,
            'commitment_loss': 0, 'adversarial_loss': 0
        }
        
        # Interleaved training as specified in paper
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i, batch in enumerate(pbar):
            scene_tensors = batch['scene'].to(DEVICE)
            pathloss_maps = batch['pathloss'].to(DEVICE)
            
            if i % 2 == 0:
                # Scene tensor batch: train encoder only
                opt_encoder.zero_grad()
                
                scene_recon, commitment_loss, _ = vqgan(scene_tensors, mode='scene')
                
                # Loss: L_rec + λ * L_commit (no adversarial for scene)
                rec_loss = l2_loss(scene_recon, scene_tensors)
                encoder_loss = rec_loss + args.commitment_weight * commitment_loss
                
                encoder_loss.backward()
                opt_encoder.step()
                
                epoch_stats['scene_batches'] += 1
                epoch_stats['encoder_loss'] += encoder_loss.item()
                epoch_stats['commitment_loss'] += commitment_loss.item()
                
            else:
                # Pathloss map batch: train decoder, discriminator
                opt_disc.zero_grad()
                pathloss_recon, _, _ = vqgan(pathloss_maps, mode='pathloss')
                real_logits = discriminator_G(pathloss_maps)
                fake_logits = discriminator_G(pathloss_recon.detach())
                
                # Cross-entropy loss for discriminator
                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(fake_logits)
                
                d_real_loss = bce_loss(real_logits, real_labels)
                d_fake_loss = bce_loss(fake_logits, fake_labels)
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                
                d_loss.backward()
                opt_disc.step()
                
                # Train decoder with adversarial loss
                opt_decoder.zero_grad()
                
                pathloss_recon, commitment_loss, _ = vqgan(pathloss_maps, mode='pathloss')
                
                # Reconstruction loss
                rec_loss = l2_loss(pathloss_recon, pathloss_maps)
                
                # Adversarial loss: L_adv = -log(G(ŷ))
                fake_logits_for_gen = discriminator_G(pathloss_recon)
                adversarial_loss = bce_loss(fake_logits_for_gen, real_labels)
                
                # Total decoder loss: L_rec + λ * L_commit + L_adv
                decoder_loss = (rec_loss + 
                               args.commitment_weight * commitment_loss + 
                               adversarial_loss)
                
                decoder_loss.backward()
                opt_decoder.step()
                
                epoch_stats['pathloss_batches'] += 1
                epoch_stats['decoder_loss'] += decoder_loss.item()
                epoch_stats['disc_loss'] += d_loss.item()
                epoch_stats['adversarial_loss'] += adversarial_loss.item()
            
            # Update progress
            pbar.set_postfix({
                'Enc': f"{epoch_stats['encoder_loss']/max(epoch_stats['scene_batches'],1):.3f}",
                'Dec': f"{epoch_stats['decoder_loss']/max(epoch_stats['pathloss_batches'],1):.3f}",
                'Disc': f"{epoch_stats['disc_loss']/max(epoch_stats['pathloss_batches'],1):.3f}"
            })
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_loss = validate_model(vqgan, val_loader, l2_loss, args)
            
            print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(vqgan, discriminator_G, epoch, val_loss, args, 'best_frozen_vqgan.pth')
                print(f"Best model saved (val_loss: {val_loss:.4f})")
        
        # Regular checkpoint
        if (epoch + 1) % 25 == 0:
            save_checkpoint(vqgan, discriminator_G, epoch, 0, args, f'frozen_vqgan_epoch_{epoch+1}.pth')
    
    print("Frozen codebook training completed!")

def validate_model(vqgan, val_loader, loss_fn, args):
    """Validation function"""
    vqgan.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            scene_tensors = batch['scene'].to(DEVICE)
            pathloss_maps = batch['pathloss'].to(DEVICE)
            
            # Validate both modes
            scene_recon, scene_commit, _ = vqgan(scene_tensors, mode='both')
            pathloss_recon, pathloss_commit, _ = vqgan(pathloss_maps, mode='both')
            
            scene_loss = loss_fn(scene_recon, scene_tensors) + args.commitment_weight * scene_commit
            pathloss_loss = loss_fn(pathloss_recon, pathloss_maps) + args.commitment_weight * pathloss_commit
            
            total_loss += (scene_loss + pathloss_loss).item()
            num_batches += 1
    
    vqgan.train()
    return total_loss / num_batches

def save_checkpoint(vqgan, discriminator, epoch, val_loss, args, filename):
    """Save checkpoint"""
    torch.save({
        'epoch': epoch,
        'vqgan_state_dict': vqgan.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'val_loss': val_loss,
        'config': {
            'embed_dim': args.embed_dim,
            'n_embed': args.n_embed,
            'ch': args.ch,
            'resolution': args.image_size,
            'codebook_frozen': True,
            'training_method': 'frozen_codebook_interleaved'
        }
    }, os.path.join(args.checkpoint_dir, filename))

def main():
    parser = argparse.ArgumentParser(description='Frozen Codebook VQGAN Pretraining')
    
    # Data paths - as specified in your request
    parser.add_argument('--scene_dir', type=str, 
                       default='.../ripple/dataset/...',
                       help='Directory containing 3-channel scene maps')
    parser.add_argument('--pathloss_dir', type=str,
                       default='.../ripple/dataset/...',
                       help='Directory containing pathloss maps')
    parser.add_argument('--pretrained_codebook_path', type=str, required=True,
                       help='Path to pretrained VQGAN checkpoint for codebook extraction')
    
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_embed', type=int, default=8192)
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (100 as per paper)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (16 as per paper)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (1e-4 as per paper)')
    parser.add_argument('--commitment_weight', type=float, default=0.25,
                       help='Commitment loss weight λ (0.25)')
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_frozen')
    
    args = parser.parse_args()
    
    print("FROZEN CODEBOOK VQGAN PRETRAINING")
    print("=" * 50)
    print("Based on paper methodology:")
    print("   Codebook: FROZEN (preserves LLaMA token space)")
    print("   Encoder: Updates on scene tensors only")
    print("   Decoder: Updates on pathloss maps only")
    print("   Discriminator G(·): Standard adversarial training")
    print("   Loss: L_rec + λ*L_commit + L_adv")
    print("   Training: Interleaved batches (scene/pathloss)")
    print("=" * 50)
    
    train_frozen_codebook_vqgan(args)

if __name__ == "__main__":

    main()
