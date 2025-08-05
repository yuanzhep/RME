import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

from frozen_codebook_pretrain import (
    RadioMapDataset, FrozenCodebookVQGAN, Discriminator,
    load_pretrained_codebook, save_checkpoint, validate_model
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SeparatedBatchTrainer:
    def __init__(self, vqgan, discriminator, args):
        self.vqgan = vqgan
        self.discriminator = discriminator
        self.args = args
        
        # Separate optimizers for conditional updates
        self.opt_encoder = optim.Adam(vqgan.encoder.parameters(), lr=args.lr, betas=(0.5, 0.9))
        self.opt_decoder = optim.Adam(vqgan.decoder.parameters(), lr=args.lr, betas=(0.5, 0.9))
        self.opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))
        
        # Loss functions
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def train_epoch_separated_batches(self, train_loader, epoch):
        self.vqgan.train()
        self.discriminator.train()
        all_batches = list(train_loader)
        total_batches = len(all_batches)
        scene_losses = []
        pathloss_losses = []
        
        print(f"Epoch {epoch+1}: Processing {total_batches} batches")
        
        # Phase 1: Process all batches for scene tensor training (encoder updates)
        print("Phase 1: Scene tensor processing (encoder updates)")
        for i, batch in enumerate(tqdm(all_batches, desc="Scene Processing")):
            scene_tensors = batch['scene'].to(DEVICE)
            
            self.opt_encoder.zero_grad()
            
            # Forward with scene tensors (encoder active, decoder frozen)
            scene_recon, commitment_loss, _ = self.vqgan(scene_tensors, mode='scene')
            
            # Loss: L_rec + λ * L_commit (only reconstruction + commitment)
            rec_loss = self.l2_loss(scene_recon, scene_tensors)
            encoder_loss = rec_loss + self.args.commitment_weight * commitment_loss
            
            encoder_loss.backward()
            self.opt_encoder.step()
            
            scene_losses.append(encoder_loss.item())
        
        # Phase 2: Process all batches for PL training (decoder updates)
        print("Phase 2: Pathloss processing (decoder + discriminator updates)")
        for i, batch in enumerate(tqdm(all_batches, desc="Pathloss Processing")):
            pathloss_maps = batch['pathloss'].to(DEVICE)
            
            # Train discriminator G(·)
            self.opt_disc.zero_grad()
            
            # Forward with pathloss maps (decoder active, encoder frozen)
            pathloss_recon, commitment_loss, _ = self.vqgan(pathloss_maps, mode='pathloss')
            real_logits = self.discriminator(pathloss_maps)
            fake_logits = self.discriminator(pathloss_recon.detach())
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            d_real_loss = self.bce_loss(real_logits, real_labels)
            d_fake_loss = self.bce_loss(fake_logits, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            
            d_loss.backward()
            self.opt_disc.step()
            
            # Train decoder with adversarial loss
            self.opt_decoder.zero_grad()
            
            # Re-forward for decoder training
            pathloss_recon, commitment_loss, _ = self.vqgan(pathloss_maps, mode='pathloss')
            
            # Reconstruction loss
            rec_loss = self.l2_loss(pathloss_recon, pathloss_maps)
            
            # Adversarial loss: L_adv = -log(G(ŷ))
            fake_logits_for_gen = self.discriminator(pathloss_recon)
            adversarial_loss = self.bce_loss(fake_logits_for_gen, real_labels)
            
            # Total decoder loss: L_rec + λ * L_commit + L_adv
            decoder_loss = (rec_loss + 
                           self.args.commitment_weight * commitment_loss + 
                           adversarial_loss)
            
            decoder_loss.backward()
            self.opt_decoder.step()
            
            pathloss_losses.append(decoder_loss.item())
        
        avg_scene_loss = sum(scene_losses) / len(scene_losses)
        avg_pathloss_loss = sum(pathloss_losses) / len(pathloss_losses)
        
        print(f"Epoch {epoch+1} completed:")
        print(f"   Scene loss (encoder): {avg_scene_loss:.4f}")
        print(f"   Pathloss loss (decoder): {avg_pathloss_loss:.4f}")
        
        return avg_scene_loss, avg_pathloss_loss

def train_with_separated_batches(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load frozen codebook
    pretrained_codebook = load_pretrained_codebook(args.pretrained_codebook_path)
    if pretrained_codebook is None:
        raise ValueError("Could not load pretrained codebook!")
    
    # Dataset
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
    
    discriminator_G = Discriminator(input_nc=3).to(DEVICE)
    
    # Trainer
    trainer = SeparatedBatchTrainer(vqgan, discriminator_G, args)
    
    print("EXPLICIT BATCH SEPARATION TRAINING")
    print("=" * 50)
    print("Training method:")
    print("   Phase 1: All batches → Scene processing (encoder updates)")
    print("   Phase 2: All batches → Pathloss processing (decoder updates)")
    print("   This ensures complete separation as per paper description")
    print("=" * 50)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train with separated batches
        scene_loss, pathloss_loss = trainer.train_epoch_separated_batches(train_loader, epoch)
        if (epoch + 1) % 10 == 0:
            val_loss = validate_model(vqgan, val_loader, trainer.l2_loss, args)
            
            print(f"Validation: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(vqgan, discriminator_G, epoch, val_loss, args, 
                               'best_separated_batch_vqgan.pth')
                print(f"Best model saved")
        
        # Regular checkpoint
        if (epoch + 1) % 25 == 0:
            save_checkpoint(vqgan, discriminator_G, epoch, 0, args, 
                           f'separated_batch_epoch_{epoch+1}.pth')
    
    print("Separated batch training completed")

def main():
    parser = argparse.ArgumentParser(description='Separated Batch VQGAN Training')
    parser.add_argument('--scene_dir', type=str, 
                       default='.../ripple/dataset/...')
    parser.add_argument('--pathloss_dir', type=str,
                       default='.../ripple/dataset/...')
    parser.add_argument('--pretrained_codebook_path', type=str, required=True)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_embed', type=int, default=8192)
    parser.add_argument('--ch', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--commitment_weight', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_separated')
    
    args = parser.parse_args()
    
    train_with_separated_batches(args)

if __name__ == "__main__":

    main()
