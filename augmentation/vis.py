import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imread, imsave
import copy
from datetime import datetime
import argparse
import random

from transforms import (
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
    RandomScale, RandomCrop, AddGaussianNoise, RandomBrightness,
    RandomContrast, RandomChannelShuffle, RandomElasticDeformation,
    Compose
)

class RepresentativeSampleVisualizer:
    def __init__(self, input_path, output_path, save_dir="representative_augmentations"):
        self.input_path = input_path
        self.output_path = output_path
        self.save_dir = save_dir
        self.cmap = copy.copy(cm.get_cmap('jet_r'))

        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
    
    def find_representative_samples(self, max_search=100):
        available_files = []
        for b in range(0, 25):  # Buildings 1-25
            for s in range(50):  # Samples 0-49
                filename = f"B{b}_Ant1_f1_S{s}.png"
                input_file = os.path.join(self.input_path, filename)
                output_file = os.path.join(self.output_path, filename)
                
                if os.path.exists(input_file) and os.path.exists(output_file):
                    available_files.append(filename)
                    if len(available_files) >= max_search:
                        break
            if len(available_files) >= max_search:
                break
        
        if not available_files:
            print("No files found!")
            return []
        
        print(f"Found {len(available_files)} available files")
        
        # Select 1-2 representative samples
        # Strategy: pick from different buildings and different sample indices
        selected = []
        
        if len(available_files) >= 1:
            early_candidates = [f for f in available_files if f.startswith('B1_') or f.startswith('B2_')]
            if early_candidates:
                selected.append(random.choice(early_candidates))
            else:
                selected.append(available_files[0])
        
        if len(available_files) >= 2:
            remaining = [f for f in available_files if f not in selected]
            if remaining:
                first_building = selected[0].split('_')[0]
                different_building = [f for f in remaining if not f.startswith(first_building)]
                if different_building:
                    selected.append(random.choice(different_building))
                else:
                    selected.append(remaining[0])
        
        print(f"Selected representative samples: {selected}")
        return selected
    
    def load_sample(self, filename):
        input_file = os.path.join(self.input_path, filename)
        output_file = os.path.join(self.output_path, filename)
        input_img = imread(input_file)
        output_img = imread(output_file)
        
        if len(input_img.shape) == 2:
            input_img = np.stack([input_img] * 3, axis=-1)
        
        if len(output_img.shape) == 3:
            output_img = output_img[:, :, 0]  # Take first channel
        
        input_img = input_img.astype(np.float32) / 255.0
        output_img = output_img.astype(np.float32) / 255.0
        
        return input_img, output_img
    
    def create_augmentation_transforms(self):
        transforms_dict = {
            'Original': None,
            'Horizontal_Flip': RandomHorizontalFlip(p=1.0),
            'Vertical_Flip': RandomVerticalFlip(p=1.0),
            'Rotation_90': RandomRotation(angles=[90], p=1.0),
            'Rotation_180': RandomRotation(angles=[180], p=1.0),
            'Scale_Up_1.2x': RandomScale(scale_range=(1.2, 1.2), p=1.0),
            'Scale_Down_0.8x': RandomScale(scale_range=(0.8, 0.8), p=1.0),
            'Noise_Light': AddGaussianNoise(noise_std=0.02, p=1.0),
            'Noise_Medium': AddGaussianNoise(noise_std=0.05, p=1.0),
            'Bright_Up': RandomBrightness(brightness_range=(1.3, 1.3), p=1.0),
            'Bright_Down': RandomBrightness(brightness_range=(0.7, 0.7), p=1.0),
            'Contrast_Up': RandomContrast(contrast_range=(1.4, 1.4), p=1.0),
            'Contrast_Down': RandomContrast(contrast_range=(0.6, 0.6), p=1.0),
            'Channel_Shuffle': RandomChannelShuffle(p=1.0),
            'Elastic_Mild': RandomElasticDeformation(alpha=30, sigma=3, p=1.0),
            'Elastic_Strong': RandomElasticDeformation(alpha=60, sigma=5, p=1.0),
            'Combo_Flip_Noise': Compose([
                RandomHorizontalFlip(p=1.0),
                AddGaussianNoise(noise_std=0.02, p=1.0)
            ]),
            'Combo_Rotate_Scale': Compose([
                RandomRotation(angles=[90], p=1.0),
                RandomScale(scale_range=(1.1, 1.1), p=1.0)
            ]),
            'Combo_Complex': Compose([
                RandomHorizontalFlip(p=1.0),
                RandomRotation(angles=[180], p=1.0),
                RandomScale(scale_range=(0.9, 0.9), p=1.0),
                AddGaussianNoise(noise_std=0.03, p=1.0)
            ])
        }
        
        return transforms_dict
    
    def apply_augmentation(self, input_img, output_img, transform):
        if transform is None:
            return input_img.copy(), output_img.copy()
        
        return transform(input_img.copy(), output_img.copy())
    
    def save_individual_augmented_images(self, input_img, output_img, sample_name, transform_name):
        sample_dir = os.path.join(self.save_dir, f"{sample_name}_individual_augmentations")
        input_dir = os.path.join(sample_dir, "inputs")
        output_dir = os.path.join(sample_dir, "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        for ch in range(input_img.shape[2]):
            input_ch = input_img[:, :, ch]
            input_ch_uint8 = (np.clip(input_ch, 0, 1) * 255).astype(np.uint8)
            filename = f"{transform_name}_input_channel_{ch+1}.png"
            imsave(os.path.join(input_dir, filename), input_ch_uint8)
        
        input_rgb = (np.clip(input_img, 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(input_dir, f"{transform_name}_input_combined.png"), input_rgb)
        
        output_uint8 = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(output_dir, f"{transform_name}_output.png"), output_uint8)
        
        return sample_dir
    
    def create_comprehensive_grid(self, input_img, output_img, sample_name):
        transforms_dict = self.create_augmentation_transforms()
        num_transforms = len(transforms_dict)
        cols = 6  # input_ch1, input_ch2, input_ch3, output, diff_map, stats
        rows = num_transforms
        fig, axes = plt.subplots(rows, cols, figsize=(24, 3 * rows))
        fig.suptitle(f'Comprehensive Augmentation Analysis - {sample_name}', fontsize=20, y=0.98)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        original_output = None
        
        for row, (transform_name, transform) in enumerate(transforms_dict.items()):
            aug_input, aug_output = self.apply_augmentation(input_img, output_img, transform)
            if transform_name == 'Original':
                original_output = aug_output.copy()

            self.save_individual_augmented_images(aug_input, aug_output, sample_name, transform_name)
            
            for ch in range(3):
                if ch < aug_input.shape[2]:
                    axes[row, ch].imshow(aug_input[:, :, ch], cmap='gray', vmin=0, vmax=1)
                    axes[row, ch].set_title(f'{transform_name}\nInput Ch{ch+1}', fontsize=10)
                else:
                    axes[row, ch].imshow(np.zeros_like(aug_input[:, :, 0]), cmap='gray')
                    axes[row, ch].set_title(f'{transform_name}\nNo Ch{ch+1}', fontsize=10)
                axes[row, ch].axis('off')
            
            im_out = axes[row, 3].imshow(aug_output, cmap=self.cmap, vmin=0, vmax=1)
            axes[row, 3].set_title(f'{transform_name}\nOutput', fontsize=10)
            axes[row, 3].axis('off')
            
            if transform_name != 'Original' and original_output is not None:
                diff = np.abs(aug_output - original_output)
                im_diff = axes[row, 4].imshow(diff, cmap='hot', vmin=0, vmax=diff.max() if diff.max() > 0 else 1)
                axes[row, 4].set_title(f'Difference Map\nMax: {diff.max():.3f}', fontsize=10)
            else:
                axes[row, 4].imshow(np.zeros_like(aug_output), cmap='hot')
                axes[row, 4].set_title('Original\n(No Difference)', fontsize=10)
            axes[row, 4].axis('off')
            
            axes[row, 5].axis('off')
            stats_text = f"Output Statistics:\n"
            stats_text += f"Min: {aug_output.min():.3f}\n"
            stats_text += f"Max: {aug_output.max():.3f}\n"
            stats_text += f"Mean: {aug_output.mean():.3f}\n"
            stats_text += f"Std: {aug_output.std():.3f}\n"
            
            if transform_name != 'Original' and original_output is not None:
                diff_mean = np.abs(aug_output - original_output).mean()
                stats_text += f"Avg Diff: {diff_mean:.3f}\n"
                
                # Correlation with original
                correlation = np.corrcoef(aug_output.flatten(), original_output.flatten())[0, 1]
                stats_text += f"Correlation: {correlation:.3f}"
            
            axes[row, 5].text(0.05, 0.95, stats_text, transform=axes[row, 5].transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        fig.colorbar(im_out, ax=axes[:, 3], shrink=0.6, aspect=30, label='Output Value')
        if 'im_diff' in locals():
            fig.colorbar(im_diff, ax=axes[:, 4], shrink=0.6, aspect=30, label='Difference')       
        plt.tight_layout()
        grid_filename = os.path.join(self.save_dir, f"{sample_name}_comprehensive_grid.png")
        plt.savefig(grid_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive grid: {grid_filename}")
        return len(transforms_dict)
    
    def create_side_by_side_comparison(self, samples_data):
        if len(samples_data) != 2:
            return
            
        key_transforms = ['Original', 'Horizontal_Flip', 'Rotation_90', 'Noise_Medium', 
                         'Bright_Up', 'Elastic_Mild', 'Combo_Complex']
        
        transforms_dict = self.create_augmentation_transforms()
        
        fig, axes = plt.subplots(len(key_transforms), 4, figsize=(16, 3 * len(key_transforms)))
        fig.suptitle('Side-by-Side Sample Comparison - Key Augmentations', fontsize=16, y=0.98)
        
        for row, transform_name in enumerate(key_transforms):
            transform = transforms_dict[transform_name]
            
            for col, (sample_name, sample_input, sample_output) in enumerate(samples_data):
                aug_input, aug_output = self.apply_augmentation(sample_input, sample_output, transform)
                axes[row, col*2].imshow(aug_input[:, :, 0], cmap='gray', vmin=0, vmax=1)
                if row == 0:
                    axes[row, col*2].set_title(f'{sample_name}\nInput', fontsize=10)
                else:
                    axes[row, col*2].set_title(f'{transform_name}\nInput', fontsize=10)
                axes[row, col*2].axis('off')
                axes[row, col*2+1].imshow(aug_output, cmap=self.cmap, vmin=0, vmax=1)
                if row == 0:
                    axes[row, col*2+1].set_title(f'{sample_name}\nOutput', fontsize=10)
                else:
                    axes[row, col*2+1].set_title(f'{transform_name}\nOutput', fontsize=10)
                axes[row, col*2+1].axis('off')
        
        plt.tight_layout()
        comparison_filename = os.path.join(self.save_dir, "samples_side_by_side_comparison.png")
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved side-by-side comparison: {comparison_filename}")
    
    def create_summary_report(self, samples_data, total_augmentations):
        report_lines = []
        report_lines.append("REPRESENTATIVE SAMPLE AUGMENTATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Input path: {self.input_path}")
        report_lines.append(f"Output path: {self.output_path}")
        report_lines.append(f"Save directory: {self.save_dir}")
        report_lines.append("")
        report_lines.append("SELECTED SAMPLES:")
        report_lines.append("-" * 20)
        for i, (sample_name, _, _) in enumerate(samples_data):
            report_lines.append(f"{i+1}. {sample_name}")
        report_lines.append("")
        report_lines.append("AUGMENTATIONS APPLIED:")
        report_lines.append("-" * 25)
        report_lines.append(f"Total augmentations: {total_augmentations}")
        transforms_dict = self.create_augmentation_transforms()
        for i, transform_name in enumerate(transforms_dict.keys()):
            report_lines.append(f"{i+1:2d}. {transform_name}")
        report_lines.append("")
        report_lines.append("GENERATED FILES:")
        report_lines.append("-" * 20)
        report_lines.append(" Comprehensive grids: *_comprehensive_grid.png")
        report_lines.append(" Individual augmentations: */individual_augmentations/")
        report_lines.append("   ├── inputs/")
        report_lines.append("   │   ├── *_input_channel_1.png")
        report_lines.append("   │   ├── *_input_channel_2.png") 
        report_lines.append("   │   ├── *_input_channel_3.png")
        report_lines.append("   │   └── *_input_combined.png")
        report_lines.append("   └── outputs/")
        report_lines.append("       └── *_output.png")
        if len(samples_data) == 2:
            report_lines.append("Side-by-side comparison: samples_side_by_side_comparison.png")
        report_lines.append("")
        report_lines.append("USAGE NOTES:")
        report_lines.append("-" * 15)
        report_lines.append("• Individual PNG files can be used directly in presentations")
        report_lines.append("• Comprehensive grids show all augmentation effects at once") 
        report_lines.append("• Difference maps highlight areas most affected by augmentations")
        report_lines.append("• Statistics help quantify the impact of each augmentation")
        
        report_filename = os.path.join(self.save_dir, "augmentation_report.txt")
        with open(report_filename, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved report: {report_filename}")
        print("\n" + "=" * 60)
        print("AUGMENTATION VISUALIZATION COMPLETE!")
        print("=" * 60)
        print(f"Results directory: {self.save_dir}")
        print(f"Samples processed: {len(samples_data)}")
        print(f"Augmentations applied: {total_augmentations}")
        print(f"Individual PNG files: {len(samples_data) * total_augmentations * 4}")  # 3 input channels + 1 output
    
    def visualize_representative_samples(self, max_samples=2):
        print("Starting representative sample augmentation visualization...")
        selected_files = self.find_representative_samples()
        if not selected_files:
            print("No valid data files found!")
            return

        if len(selected_files) > max_samples:
            selected_files = selected_files[:max_samples]
        
        samples_data = []
        for filename in selected_files:
            try:
                input_img, output_img = self.load_sample(filename)
                sample_name = filename.replace('.png', '')
                samples_data.append((sample_name, input_img, output_img))
                print(f"Loaded sample: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        if not samples_data:
            print("No samples could be loaded!")
            return
        
        total_augmentations = 0
        for sample_name, input_img, output_img in samples_data:
            print(f"\nProcessing {sample_name}...")
            num_augs = self.create_comprehensive_grid(input_img, output_img, sample_name)
            total_augmentations = num_augs
        
        if len(samples_data) == 2:
            print("\nCreating side-by-side comparison...")
            self.create_side_by_side_comparison(samples_data)
        
        print("\nGenerating summary report...")
        self.create_summary_report(samples_data, total_augmentations)


def main():
    parser = argparse.ArgumentParser(description='Visualize augmentations on representative radio map samples')
    parser.add_argument('--input_path', type=str, 
                       default=".../Dataset/Inputs/...",
                       help='Input data path')
    parser.add_argument('--output_path', type=str,
                       default=".../Dataset/Outputs/...", 
                       help='Output data path')
    parser.add_argument('--save_dir', type=str, default="representative_augmentations",
                       help='Directory to save results')
    parser.add_argument('--max_samples', type=int, default=2, choices=[1, 2],
                       help='Maximum number of representative samples (1 or 2)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible sample selection')
    
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    if not os.path.exists(args.input_path):
        print(f"Input path does not exist: {args.input_path}")
        return
    
    if not os.path.exists(args.output_path):
        print(f"Output path does not exist: {args.output_path}")
        return
    
    visualizer = RepresentativeSampleVisualizer(args.input_path, args.output_path, args.save_dir)
    visualizer.visualize_representative_samples(max_samples=args.max_samples)

if __name__ == "__main__":
    main()
