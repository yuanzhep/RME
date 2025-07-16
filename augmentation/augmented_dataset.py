"""
Enhanced Dataset Generator with Data Augmentation for VQGAN Pretraining
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import cv2
import os
import random
from typing import Optional, Tuple, List
from transforms import Compose, Resize, Normalize


class AugmentedRadioMapDataset(Dataset):
    """
    Enhanced Dataset for Radio Map Prediction with Data Augmentation
    Supports VQGAN pretraining with encoder-decoder updates
    """
    
    def __init__(self, 
                 input_path: str,
                 output_path: str,
                 buildings: List[int] = None,
                 antennas: List[int] = [1],
                 frequencies: List[int] = [1],
                 samples_per_config: int = 50,
                 image_size: Tuple[int, int] = (512, 512),
                 transforms: Optional[Compose] = None,
                 device: str = "cuda",
                 normalize_input: bool = True,
                 data_format: str = "CHW"):  # "CHW" or "HWC"
        """
        Initialize the augmented dataset
        
        Args:
            input_path: Path to input images (3-channel scene graphs)
            output_path: Path to output images (grayscale pathloss maps)
            buildings: List of building IDs to include (default: 1-25)
            antennas: List of antenna IDs to include (default: [1])
            frequencies: List of frequency IDs to include (default: [1])
            samples_per_config: Number of samples per configuration (default: 50)
            image_size: Target image size (H, W)
            transforms: Data augmentation transforms
            device: Device to load tensors on
            normalize_input: Whether to normalize input to [0, 1]
            data_format: Output tensor format - "CHW" (channels first) or "HWC" (channels last)
        """
        
        self.input_path = input_path
        self.output_path = output_path
        self.image_size = image_size
        self.transforms = transforms
        self.device = device
        self.normalize_input = normalize_input
        self.data_format = data_format
        
        # Default parameters
        if buildings is None:
            buildings = list(range(1, 26))  # Buildings 1-25
        
        self.buildings = buildings
        self.antennas = antennas
        self.frequencies = frequencies
        self.samples_per_config = samples_per_config
        
        # Generate file names
        self.file_names = []
        self.file_indices = []
        
        for b in buildings:
            for a in antennas:
                for f in frequencies:
                    for s in range(samples_per_config):
                        filename = f"B{b}_Ant{a}_f{f}_S{s}"
                        self.file_names.append(filename)
                        self.file_indices.append(len(self.file_names) - 1)
        
        print(f"Dataset initialized with {len(self.file_names)} samples")
        print(f"Buildings: {buildings}")
        print(f"Antennas: {antennas}")
        print(f"Frequencies: {frequencies}")
        print(f"Samples per config: {samples_per_config}")
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        """Get a single sample with augmentation"""
        
        filename = self.file_names[idx]
        
        # Load input image (3-channel scene graph)
        input_path = os.path.join(self.input_path, filename + ".png")
        input_img = imread(input_path)
        
        # Load output image (grayscale pathloss map)
        output_path = os.path.join(self.output_path, filename + ".png")
        output_img = imread(output_path)
        
        # Ensure correct dimensions
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        if len(output_img.shape) == 3:
            output_img = output_img[:, :, 0]  # Take first channel if RGB
        
        # Apply transforms if provided
        if self.transforms:
            input_img, output_img = self.transforms(input_img, output_img)
        
        # Resize to target size
        input_img = cv2.resize(input_img, self.image_size, interpolation=cv2.INTER_NEAREST)
        output_img = cv2.resize(output_img, self.image_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize input to [0, 1]
        if self.normalize_input:
            input_img = input_img.astype(np.float32) / 255.0
        
        # Normalize output to [0, 1] if needed
        if output_img.dtype == np.uint8:
            output_img = output_img.astype(np.float32) / 255.0
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_img).float()
        output_tensor = torch.from_numpy(output_img).float()
        
        # Adjust tensor format
        if self.data_format == "CHW":
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.permute(2, 0, 1)  # HWC -> CHW
            if len(output_tensor.shape) == 2:
                output_tensor = output_tensor.unsqueeze(0)  # HW -> 1HW
        
        return input_tensor, output_tensor, filename
    
    def get_sample_info(self, idx):
        """Get information about a specific sample"""
        filename = self.file_names[idx]
        parts = filename.split('_')
        
        building = int(parts[0][1:])  # Extract building number
        antenna = int(parts[1][3:])   # Extract antenna number
        freq = int(parts[2][1:])      # Extract frequency number
        sample = int(parts[3][1:])    # Extract sample number
        
        return {
            'building': building,
            'antenna': antenna,
            'frequency': freq,
            'sample': sample,
            'filename': filename
        }


class DataAugmentationPresets:
    """Predefined augmentation presets for different training scenarios"""
    
    @staticmethod
    def get_light_augmentation():
        """Light augmentation for initial training"""
        from transforms import (RandomHorizontalFlip, RandomVerticalFlip, 
                              RandomRotation, AddGaussianNoise, Compose)
        
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(p=0.3),
            AddGaussianNoise(noise_std=0.01, p=0.3),
        ])
    
    @staticmethod
    def get_medium_augmentation():
        """Medium augmentation for robust training"""
        from transforms import (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
                              RandomScale, AddGaussianNoise, RandomBrightness, 
                              RandomContrast, Compose)
        
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(p=0.4),
            RandomScale(scale_range=(0.9, 1.1), p=0.3),
            AddGaussianNoise(noise_std=0.02, p=0.4),
            RandomBrightness(brightness_range=(0.9, 1.1), p=0.3),
            RandomContrast(contrast_range=(0.9, 1.1), p=0.3),
        ])
    
    @staticmethod
    def get_heavy_augmentation():
        """Heavy augmentation for maximum robustness"""
        from transforms import (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
                              RandomScale, RandomCrop, AddGaussianNoise, RandomBrightness,
                              RandomContrast, RandomChannelShuffle, RandomElasticDeformation,
                              Compose)
        
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(p=0.5),
            RandomScale(scale_range=(0.8, 1.2), p=0.4),
            RandomCrop(size=(480, 480), p=0.3),  # Crop then will be resized
            AddGaussianNoise(noise_std=0.03, p=0.4),
            RandomBrightness(brightness_range=(0.8, 1.2), p=0.4),
            RandomContrast(contrast_range=(0.8, 1.2), p=0.4),
            RandomChannelShuffle(p=0.2),
            RandomElasticDeformation(alpha=50, sigma=5, p=0.2),
        ])
    
    @staticmethod
    def get_vqgan_pretraining_augmentation():
        """Optimized augmentation for VQGAN pretraining"""
        from transforms import (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
                              RandomScale, AddGaussianNoise, RandomBrightness, 
                              RandomContrast, Compose)
        
        return Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(angles=[90, 180, 270], p=0.3),
            RandomScale(scale_range=(0.95, 1.05), p=0.2),  # Mild scaling
            AddGaussianNoise(noise_std=0.015, p=0.3),
            RandomBrightness(brightness_range=(0.95, 1.05), p=0.2),
            RandomContrast(contrast_range=(0.95, 1.05), p=0.2),
        ])


def create_dataloaders(input_path: str,
                      output_path: str,
                      batch_size: int = 8,
                      train_split: float = 0.8,
                      val_split: float = 0.1,
                      test_split: float = 0.1,
                      augmentation_preset: str = "vqgan",
                      image_size: Tuple[int, int] = (512, 512),
                      num_workers: int = 4,
                      buildings: List[int] = None,
                      device: str = "cuda") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        input_path: Path to input images
        output_path: Path to output images
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        augmentation_preset: Augmentation preset ("light", "medium", "heavy", "vqgan")
        image_size: Target image size
        num_workers: Number of workers for data loading
        buildings: List of building IDs to include
        device: Device to load tensors on
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Get augmentation transforms
    if augmentation_preset == "light":
        train_transforms = DataAugmentationPresets.get_light_augmentation()
    elif augmentation_preset == "medium":
        train_transforms = DataAugmentationPresets.get_medium_augmentation()
    elif augmentation_preset == "heavy":
        train_transforms = DataAugmentationPresets.get_heavy_augmentation()
    elif augmentation_preset == "vqgan":
        train_transforms = DataAugmentationPresets.get_vqgan_pretraining_augmentation()
    else:
        train_transforms = None
    
    # No augmentation for validation and test
    val_test_transforms = None
    
    # Create full dataset
    full_dataset = AugmentedRadioMapDataset(
        input_path=input_path,
        output_path=output_path,
        buildings=buildings,
        image_size=image_size,
        transforms=None,  # Will be set per split
        device=device
    )
    
    # Split indices
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Random split
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets for each split
    train_dataset = AugmentedRadioMapDataset(
        input_path=input_path,
        output_path=output_path,
        buildings=buildings,
        image_size=image_size,
        transforms=train_transforms,
        device=device
    )
    
    val_dataset = AugmentedRadioMapDataset(
        input_path=input_path,
        output_path=output_path,
        buildings=buildings,
        image_size=image_size,
        transforms=val_test_transforms,
        device=device
    )
    
    test_dataset = AugmentedRadioMapDataset(
        input_path=input_path,
        output_path=output_path,
        buildings=buildings,
        image_size=image_size,
        transforms=val_test_transforms,
        device=device
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches ({len(train_indices)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_indices)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_indices)} samples)")
    
    return train_loader, val_loader, test_loader