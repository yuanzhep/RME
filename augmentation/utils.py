
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import torch
from torch.utils.data import DataLoader
import copy
from typing import Tuple, Optional, List
import os

class RadioMapVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.cmap = copy.copy(cm.get_cmap('jet_r'))
        plt.style.use('default')
    
    def plot_sample(self, input_img: np.ndarray, output_img: np.ndarray, 
                   title: str = "Radio Map Sample", save_path: Optional[str] = None):
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        if len(input_img.shape) == 3:
            for i in range(min(3, input_img.shape[2])):
                row = i // 2
                col = i % 2
                axes[row, col].imshow(input_img[:, :, i], cmap='gray')
                axes[row, col].set_title(f'Input Channel {i+1}')
                axes[row, col].axis('off')
        
        if len(output_img.shape) == 2:
            im = axes[1, 1].imshow(output_img, cmap=self.cmap)
            axes[1, 1].set_title('Output (Pathloss Map)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_augmentation_comparison(self, original_input: np.ndarray, original_output: np.ndarray,
                                   augmented_input: np.ndarray, augmented_output: np.ndarray,
                                   title: str = "Augmentation Comparison", save_path: Optional[str] = None):
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(title, fontsize=16)
        
        axes[0, 0].imshow(original_input[:, :, 0] if len(original_input.shape) == 3 else original_input, cmap='gray')
        axes[0, 0].set_title('Original Input Ch1')
        axes[0, 0].axis('off')
        
        if len(original_input.shape) == 3 and original_input.shape[2] > 1:
            axes[0, 1].imshow(original_input[:, :, 1], cmap='gray')
            axes[0, 1].set_title('Original Input Ch2')
            axes[0, 1].axis('off')
            
            if original_input.shape[2] > 2:
                axes[0, 2].imshow(original_input[:, :, 2], cmap='gray')
                axes[0, 2].set_title('Original Input Ch3')
                axes[0, 2].axis('off')
        
        im1 = axes[0, 3].imshow(original_output, cmap=self.cmap)
        axes[0, 3].set_title('Original Output')
        axes[0, 3].axis('off')
        plt.colorbar(im1, ax=axes[0, 3], shrink=0.8)
        
        axes[1, 0].imshow(augmented_input[:, :, 0] if len(augmented_input.shape) == 3 else augmented_input, cmap='gray')
        axes[1, 0].set_title('Augmented Input Ch1')
        axes[1, 0].axis('off')
        
        if len(augmented_input.shape) == 3 and augmented_input.shape[2] > 1:
            axes[1, 1].imshow(augmented_input[:, :, 1], cmap='gray')
            axes[1, 1].set_title('Augmented Input Ch2')
            axes[1, 1].axis('off')
            
            if augmented_input.shape[2] > 2:
                axes[1, 2].imshow(augmented_input[:, :, 2], cmap='gray')
                axes[1, 2].set_title('Augmented Input Ch3')
                axes[1, 2].axis('off')
        
        im2 = axes[1, 3].imshow(augmented_output, cmap=self.cmap)
        axes[1, 3].set_title('Augmented Output')
        axes[1, 3].axis('off')
        plt.colorbar(im2, ax=axes[1, 3], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_batch_samples(self, dataloader: DataLoader, num_samples: int = 4,
                          title: str = "Batch Samples", save_path: Optional[str] = None):
        
        data_iter = iter(dataloader)
        inputs, outputs, filenames = next(data_iter)
        
        if torch.is_tensor(inputs):
            inputs = inputs.cpu().numpy()
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().numpy()
        
        num_samples = min(num_samples, inputs.shape[0])
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        fig.suptitle(title, fontsize=16)
        
        for i in range(num_samples):
            input_img = inputs[i]
            output_img = outputs[i]
            
            if len(input_img.shape) == 3 and input_img.shape[0] == 3:  
                input_img = np.transpose(input_img, (1, 2, 0)) 
            if len(output_img.shape) == 3 and output_img.shape[0] == 1:  
                output_img = output_img[0] 
            
            for ch in range(min(3, input_img.shape[2] if len(input_img.shape) == 3 else 1)):
                if len(input_img.shape) == 3:
                    axes[i, ch].imshow(input_img[:, :, ch], cmap='gray')
                else:
                    axes[i, ch].imshow(input_img, cmap='gray')
                axes[i, ch].set_title(f'Sample {i+1} - Ch{ch+1}')
                axes[i, ch].axis('off')
            
            im = axes[i, 3].imshow(output_img, cmap=self.cmap)
            axes[i, 3].set_title(f'Sample {i+1} - Output')
            axes[i, 3].axis('off')
            
            if i == 0: 
                plt.colorbar(im, ax=axes[i, 3], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_augmentation_grid(self, dataset, sample_idx: int, transforms_list: List,
                              transform_names: List[str], save_path: Optional[str] = None):
        original_input, original_output, filename = dataset[sample_idx]
        
        if torch.is_tensor(original_input):
            original_input = original_input.cpu().numpy()
        if torch.is_tensor(original_output):
            original_output = original_output.cpu().numpy()
        
        if len(original_input.shape) == 3 and original_input.shape[0] == 3:  
            original_input = np.transpose(original_input, (1, 2, 0))  
        if len(original_output.shape) == 3 and original_output.shape[0] == 1: 
            original_output = original_output[0]
        
        num_transforms = len(transforms_list)
        cols = 4 
        rows = num_transforms + 1 
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(f'Augmentation Effects - {filename}', fontsize=16)
        
        self._plot_sample_row(axes[0], original_input, original_output, "Original")
        
        for i, (transform, name) in enumerate(zip(transforms_list, transform_names)):
            aug_input, aug_output = transform(original_input.copy(), original_output.copy())
            self._plot_sample_row(axes[i + 1], aug_input, aug_output, name)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_sample_row(self, axes_row, input_img, output_img, title_prefix):
        for ch in range(min(3, input_img.shape[2] if len(input_img.shape) == 3 else 1)):
            if len(input_img.shape) == 3:
                axes_row[ch].imshow(input_img[:, :, ch], cmap='gray')
            else:
                axes_row[ch].imshow(input_img, cmap='gray')
            axes_row[ch].set_title(f'{title_prefix} - Ch{ch+1}')
            axes_row[ch].axis('off')
        
        im = axes_row[3].imshow(output_img, cmap=self.cmap)
        axes_row[3].set_title(f'{title_prefix} - Output')
        axes_row[3].axis('off')
    
    def create_augmentation_summary(self, dataloader: DataLoader, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.plot_batch_samples(
            dataloader, 
            num_samples=8, 
            title="Augmented Batch Samples",
            save_path=os.path.join(output_dir, "batch_samples.png")
        )
        
        print(f"Augmentation summary saved to {output_dir}")


def test_augmentations():
    from transforms import (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
                          RandomScale, AddGaussianNoise, Compose)
    from augmented_dataset import AugmentedRadioMapDataset
    
    transforms = Compose([
        RandomHorizontalFlip(p=1.0),  # Always apply for testing
        RandomVerticalFlip(p=1.0),
        RandomRotation(p=1.0),
        AddGaussianNoise(p=1.0),
    ])
    
    test_input = np.random.rand(100, 100, 3) * 255
    test_output = np.random.rand(100, 100) * 255
    
    aug_input, aug_output = transforms(test_input, test_output)
    
    visualizer = RadioMapVisualizer()
    visualizer.plot_augmentation_comparison(
        test_input, test_output, 
        aug_input, aug_output,
        title="Augmentation Test"
    )
    
    print("Augmentation test completed")


if __name__ == "__main__":
    test_augmentations()
