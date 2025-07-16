"""
Test script to verify data augmentation pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys

# Add current directory to path to import our modules
sys.path.append('.')

from transforms import (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
                       RandomScale, AddGaussianNoise, RandomBrightness, RandomContrast,
                       RandomChannelShuffle, RandomElasticDeformation, Compose)
from augmented_dataset import AugmentedRadioMapDataset, DataAugmentationPresets, create_dataloaders
from visualization_utils import RadioMapVisualizer


def test_individual_transforms():
    """Test individual transforms with synthetic data"""
    print("Testing individual transforms...")
    
    # Create synthetic test data
    np.random.seed(42)
    test_input = np.random.rand(100, 100, 3).astype(np.float32)
    test_output = np.random.rand(100, 100).astype(np.float32)
    
    # Test transforms
    transforms_to_test = [
        ("Horizontal Flip", RandomHorizontalFlip(p=1.0)),
        ("Vertical Flip", RandomVerticalFlip(p=1.0)),
        ("Rotation", RandomRotation(p=1.0)),
        ("Scale", RandomScale(scale_range=(1.2, 1.2), p=1.0)),
        ("Gaussian Noise", AddGaussianNoise(noise_std=0.1, p=1.0)),
        ("Brightness", RandomBrightness(brightness_range=(1.5, 1.5), p=1.0)),
        ("Contrast", RandomContrast(contrast_range=(1.5, 1.5), p=1.0)),
        ("Channel Shuffle", RandomChannelShuffle(p=1.0)),
        ("Elastic Deformation", RandomElasticDeformation(alpha=20, sigma=3, p=1.0)),
    ]
    
    visualizer = RadioMapVisualizer(figsize=(20, 12))
    
    for name, transform in transforms_to_test:
        print(f"  Testing {name}...")
        
        try:
            # Apply transform
            aug_input, aug_output = transform(test_input.copy(), test_output.copy())
            
            # Visualize
            visualizer.plot_augmentation_comparison(
                test_input, test_output,
                aug_input, aug_output,
                title=f"{name} Augmentation Test"
            )
            
            print(f"    ✓ {name} test passed")
            
        except Exception as e:
            print(f"    ✗ {name} test failed: {str(e)}")
    
    print("Individual transform tests completed!\n")


def test_composed_transforms():
    """Test composed transforms"""
    print("Testing composed transforms...")
    
    # Create test data
    np.random.seed(42)
    test_input = np.random.rand(100, 100, 3).astype(np.float32)
    test_output = np.random.rand(100, 100).astype(np.float32)
    
    # Test different presets
    presets = {
        "Light": DataAugmentationPresets.get_light_augmentation(),
        "Medium": DataAugmentationPresets.get_medium_augmentation(), 
        "Heavy": DataAugmentationPresets.get_heavy_augmentation(),
        "VQGAN": DataAugmentationPresets.get_vqgan_pretraining_augmentation(),
    }
    
    visualizer = RadioMapVisualizer(figsize=(20, 15))
    
    for name, preset in presets.items():
        print(f"  Testing {name} preset...")
        
        try:
            # Apply transforms multiple times to see variation
            augmented_samples = []
            for i in range(3):
                aug_input, aug_output = preset(test_input.copy(), test_output.copy())
                augmented_samples.append((aug_input, aug_output))
            
            # Visualize first sample
            visualizer.plot_augmentation_comparison(
                test_input, test_output,
                augmented_samples[0][0], augmented_samples[0][1],
                title=f"{name} Preset Augmentation Test"
            )
            
            print(f"    ✓ {name} preset test passed")
            
        except Exception as e:
            print(f"    ✗ {name} preset test failed: {str(e)}")
    
    print("Composed transform tests completed!\n")


def test_dataset_with_synthetic_data():
    """Test dataset with synthetic data"""
    print("Testing dataset with synthetic data...")
    
    # Create a temporary directory structure for testing
    temp_dir = "./temp_test_data"
    input_dir = os.path.join(temp_dir, "inputs")
    output_dir = os.path.join(temp_dir, "outputs")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create synthetic images
        from skimage.io import imsave
        
        for b in range(1, 3):  # 2 buildings
            for s in range(5):  # 5 samples each
                filename = f"B{b}_Ant1_f1_S{s}.png"
                
                # Create synthetic input (3-channel)
                input_img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
                imsave(os.path.join(input_dir, filename), input_img)
                
                # Create synthetic output (grayscale)
                output_img = (np.random.rand(128, 128) * 255).astype(np.uint8)
                imsave(os.path.join(output_dir, filename), output_img)
        
        # Test dataset
        dataset = AugmentedRadioMapDataset(
            input_path=input_dir,
            output_path=output_dir,
            buildings=[1, 2],
            samples_per_config=5,
            image_size=(64, 64),
            transforms=DataAugmentationPresets.get_light_augmentation(),
            device="cpu"
        )
        
        print(f"  Dataset created with {len(dataset)} samples")
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            input_tensor, output_tensor, filename = dataset[i]
            print(f"    Sample {i}: {filename}")
            print(f"      Input shape: {input_tensor.shape}")
            print(f"      Output shape: {output_tensor.shape}")
            print(f"      Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            print(f"      Output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch_inputs, batch_outputs, batch_filenames in dataloader:
            print(f"  Batch test:")
            print(f"    Batch input shape: {batch_inputs.shape}")
            print(f"    Batch output shape: {batch_outputs.shape}")
            print(f"    Filenames: {batch_filenames}")
            break
        
        print("    ✓ Dataset test passed")
        
    except Exception as e:
        print(f"    ✗ Dataset test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("Dataset test completed!\n")


def test_with_real_data(input_path, output_path):
    """Test with real radio map data if available"""
    print(f"Testing with real data...")
    print(f"  Input path: {input_path}")
    print(f"  Output path: {output_path}")
    
    if not os.path.exists(input_path) or not os.path.exists(output_path):
        print("  ⚠ Real data paths not found, skipping real data test")
        return
    
    try:
        # Test with a small subset
        dataset = AugmentedRadioMapDataset(
            input_path=input_path,
            output_path=output_path,
            buildings=[1],  # Just one building for testing
            samples_per_config=5,  # Just 5 samples
            image_size=(256, 256),
            transforms=DataAugmentationPresets.get_vqgan_pretraining_augmentation(),
            device="cpu"
        )
        
        print(f"  Real dataset created with {len(dataset)} samples")
        
        # Test a sample
        if len(dataset) > 0:
            input_tensor, output_tensor, filename = dataset[0]
            print(f"    Sample test: {filename}")
            print(f"      Input shape: {input_tensor.shape}")
            print(f"      Output shape: {output_tensor.shape}")
            print(f"      Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            print(f"      Output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
            
            # Visualize if matplotlib is available
            try:
                visualizer = RadioMapVisualizer()
                
                # Convert tensor to numpy for visualization
                if len(input_tensor.shape) == 3 and input_tensor.shape[0] == 3:  # CHW
                    input_np = input_tensor.permute(1, 2, 0).numpy()
                else:
                    input_np = input_tensor.numpy()
                
                if len(output_tensor.shape) == 3 and output_tensor.shape[0] == 1:  # 1HW
                    output_np = output_tensor[0].numpy()
                else:
                    output_np = output_tensor.numpy()
                
                visualizer.plot_sample(
                    input_np, output_np,
                    title=f"Real Data Sample: {filename}"
                )
                
            except Exception as viz_e:
                print(f"    Visualization failed: {str(viz_e)}")
        
        print("    ✓ Real data test passed")
        
    except Exception as e:
        print(f"    ✗ Real data test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Real data test completed!\n")


def test_dataloader_creation():
    """Test dataloader creation function"""
    print("Testing dataloader creation...")
    
    # This will test with synthetic data since real paths might not exist
    temp_dir = "./temp_test_data"
    input_dir = os.path.join(temp_dir, "inputs")
    output_dir = os.path.join(temp_dir, "outputs")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create minimal synthetic data
        from skimage.io import imsave
        
        for b in range(1, 3):
            for s in range(10):
                filename = f"B{b}_Ant1_f1_S{s}.png"
                input_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
                output_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
                imsave(os.path.join(input_dir, filename), input_img)
                imsave(os.path.join(output_dir, filename), output_img)
        
        # Test dataloader creation
        train_loader, val_loader, test_loader = create_dataloaders(
            input_path=input_dir,
            output_path=output_dir,
            batch_size=4,
            buildings=[1, 2],
            augmentation_preset="light",
            image_size=(64, 64),
            num_workers=0,  # Avoid multiprocessing issues in testing
            device="cpu"
        )
        
        print(f"  Created dataloaders:")
        print(f"    Train: {len(train_loader)} batches")
        print(f"    Val: {len(val_loader)} batches") 
        print(f"    Test: {len(test_loader)} batches")
        
        # Test a batch from each
        for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
            if len(loader) > 0:
                batch_inputs, batch_outputs, batch_filenames = next(iter(loader))
                print(f"    {name} batch shape: {batch_inputs.shape} -> {batch_outputs.shape}")
        
        print("    ✓ Dataloader creation test passed")
        
    except Exception as e:
        print(f"    ✗ Dataloader creation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("Dataloader creation test completed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("RADIO MAP DATA AUGMENTATION TEST SUITE")
    print("=" * 60)
    
    # Test 1: Individual transforms
    test_individual_transforms()
    
    # Test 2: Composed transforms
    test_composed_transforms()
    
    # Test 3: Dataset with synthetic data
    test_dataset_with_synthetic_data()
    
    # Test 4: Dataloader creation
    test_dataloader_creation()
    
    # Test 5: Real data (if available)
    real_input_path = "/blue/jie.xu/pengy1/AR_RM_backup/ICASSP2025_Dataset/Inputs/Task_1_ICASSP"
    real_output_path = "/blue/jie.xu/pengy1/AR_RM_backup/ICASSP2025_Dataset/Outputs/Task_1_ICASSP"
    test_with_real_data(real_input_path, real_output_path)
    
    print("=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    
    print("\nIf all tests passed, your augmentation pipeline is ready!")
    print("You can now run the training script with:")
    print("  python train_vqgan_pretraining.py --config config.yaml")


if __name__ == "__main__":
    main()