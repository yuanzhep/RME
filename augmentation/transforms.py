"""
Data Augmentation for Pretrain
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import random
from typing import Tuple, Optional, Union


class RadioMapTransform:
    """Base class for radio map transformations that apply to both input and output"""
    
    def __init__(self):
        pass
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class RandomHorizontalFlip(RadioMapTransform):
    """Randomly flip both input and output horizontally with given probability"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            input_img = np.fliplr(input_img)
            output_img = np.fliplr(output_img)
        return input_img, output_img


class RandomVerticalFlip(RadioMapTransform):
    """Randomly flip both input and output vertically with given probability"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            input_img = np.flipud(input_img)
            output_img = np.flipud(output_img)
        return input_img, output_img


class RandomRotation(RadioMapTransform):
    """Randomly rotate both input and output by 90, 180, or 270 degrees"""
    
    def __init__(self, angles: list = [90, 180, 270], p: float = 0.5):
        self.angles = angles
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            angle = random.choice(self.angles)
            k = angle // 90
            input_img = np.rot90(input_img, k)
            output_img = np.rot90(output_img, k)
        return input_img, output_img


class RandomCrop(RadioMapTransform):
    """Randomly crop both input and output to specified size"""
    
    def __init__(self, size: Union[int, Tuple[int, int]], p: float = 0.5):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            h, w = input_img.shape[:2]
            new_h, new_w = self.size
            
            if h <= new_h and w <= new_w:
                return input_img, output_img
            
            top = random.randint(0, h - new_h) if h > new_h else 0
            left = random.randint(0, w - new_w) if w > new_w else 0
            
            input_img = input_img[top:top + new_h, left:left + new_w]
            output_img = output_img[top:top + new_h, left:left + new_w]
        
        return input_img, output_img


class RandomScale(RadioMapTransform):
    """Randomly scale both input and output"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            h, w = input_img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            input_img = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            output_img = cv2.resize(output_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return input_img, output_img


class AddGaussianNoise(RadioMapTransform):
    """Add Gaussian noise to input only (not to output pathloss map)"""
    
    def __init__(self, noise_std: float = 0.01, p: float = 0.5):
        self.noise_std = noise_std
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_std, input_img.shape)
            input_img = np.clip(input_img + noise, 0, 1)
        return input_img, output_img


class RandomBrightness(RadioMapTransform):
    """Randomly adjust brightness of input only"""
    
    def __init__(self, brightness_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.brightness_range = brightness_range
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            input_img = np.clip(input_img * brightness, 0, 1)
        return input_img, output_img


class RandomContrast(RadioMapTransform):
    """Randomly adjust contrast of input only"""
    
    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
            mean = np.mean(input_img)
            input_img = np.clip((input_img - mean) * contrast + mean, 0, 1)
        return input_img, output_img


class RandomChannelShuffle(RadioMapTransform):
    """Randomly shuffle channels of input (for 3-channel scene graphs)"""
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p and len(input_img.shape) == 3:
            channels = list(range(input_img.shape[2]))
            random.shuffle(channels)
            input_img = input_img[:, :, channels]
        return input_img, output_img


class RandomElasticDeformation(RadioMapTransform):
    """Apply elastic deformation to both input and output"""
    
    def __init__(self, alpha: float = 100, sigma: float = 10, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            h, w = input_img.shape[:2]
            
            # Generate random displacement fields
            dx = np.random.randn(h, w) * self.sigma
            dy = np.random.randn(h, w) * self.sigma
            
            # Apply Gaussian filter to smooth the displacement
            dx = cv2.GaussianBlur(dx, (0, 0), self.sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), self.sigma)
            
            # Scale by alpha
            dx *= self.alpha
            dy *= self.alpha
            
            # Create meshgrid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = np.clip(x + dx, 0, w - 1)
            y_new = np.clip(y + dy, 0, h - 1)
            
            # Apply deformation
            if len(input_img.shape) == 3:
                input_img = cv2.remap(input_img, x_new.astype(np.float32), y_new.astype(np.float32), 
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            else:
                input_img = cv2.remap(input_img, x_new.astype(np.float32), y_new.astype(np.float32), 
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            
            output_img = cv2.remap(output_img, x_new.astype(np.float32), y_new.astype(np.float32), 
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        return input_img, output_img


class Compose:
    """Compose multiple transformations"""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            input_img, output_img = transform(input_img, output_img)
        return input_img, output_img


class Resize(RadioMapTransform):
    """Resize both input and output to specified size"""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_img = cv2.resize(input_img, self.size, interpolation=cv2.INTER_NEAREST)
        output_img = cv2.resize(output_img, self.size, interpolation=cv2.INTER_CUBIC)
        return input_img, output_img


class Normalize(RadioMapTransform):
    """Normalize input to [0, 1] range"""
    
    def __init__(self, input_range: Tuple[float, float] = (0, 255)):
        self.input_range = input_range
    
    def __call__(self, input_img: np.ndarray, output_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_img = (input_img - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        input_img = np.clip(input_img, 0, 1)
        return input_img, output_img
