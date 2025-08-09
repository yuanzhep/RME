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
                 data_format: str = "CHW"):  
        
        self.input_path = input_path
        self.output_path = output_path
        self.image_size = image_size
        self.transforms = transforms
        self.device = device
        self.normalize_input = normalize_input
        self.data_format = data_format
        
        if buildings is None:
            buildings = list(range(1, 26))  # Buildings 1-25
        
        self.buildings = buildings
        self.antennas = antennas
        self.frequencies = frequencies
        self.samples_per_config = samples_per_config
        
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
        
        filename = self.file_names[idx]
        input_path = os.path.join(self.input_path, filename + ".png")
        input_img = imread(input_path)
        output_path = os.path.join(self.output_path, filename + ".png")
        output_img = imread(output_path)
        
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        if len(output_img.shape) == 3:
            output_img = output_img[:, :, 0]  # Take first channel if RGB
        
        if self.transforms:
            input_img, output_img = self.transforms(input_img, output_img)
        
        input_img = cv2.resize(input_img, self.image_size, interpolation=cv2.INTER_NEAREST)
        output_img = cv2.resize(output_img, self.image_size, interpolation=cv2.INTER_CUBIC)
        
        if self.normalize_input:
            input_img = input_img.astype(np.float32) / 255.0
        
        if output_img.dtype == np.uint8:
            output_img = output_img.astype(np.float32) / 255.0
        
        input_tensor = torch.from_numpy(input_img).float()
        output_tensor = torch.from_numpy(output_img).float()
        
        if self.data_format == "CHW":
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.permute(2, 0, 1) 
            if len(output_tensor.shape) == 2:
                output_tensor = output_tensor.unsqueeze(0)  
        
        return input_tensor, output_tensor, filename
    
    def get_sample_info(self, idx):
        filename = self.file_names[idx]
        parts = filename.split('_')
        building = int(parts[0][1:]) 
        antenna = int(parts[1][3:])   
        freq = int(parts[2][1:])      
        sample = int(parts[3][1:])    
        
        return {
            'building': building,
            'antenna': antenna,
            'frequency': freq,
            'sample': sample,
            'filename': filename
        }

class DataAugmentationPresets:
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
    
    val_test_transforms = None
    
    full_dataset = AugmentedRadioMapDataset(
        input_path=input_path,
        output_path=output_path,
        buildings=buildings,
        image_size=image_size,
        transforms=None,  # Will be set per split
        device=device
    )
    
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
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # dataloaders
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
