#!/bin/bash

# Setup and Run Script for VQGAN Radio Map Pretraining

echo "=========================================="
echo "VQGAN Radio Map Pretraining Setup"
echo "=========================================="

# Set error handling
set -e

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p temp

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Setting up environment..."
    
    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "vqgan_radio"; then
        echo "Creating new conda environment: vqgan_radio"
        conda create -n vqgan_radio python=3.8 -y
    fi
    
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate vqgan_radio
    
    # Install dependencies
    echo "Installing dependencies..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install scikit-image matplotlib pyyaml tqdm opencv-python pandas numpy
    
else
    echo "Conda not found. Please install dependencies manually:"
    echo "  pip install torch torchvision torchaudio"
    echo "  pip install scikit-image matplotlib pyyaml tqdm opencv-python pandas numpy"
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Testing the augmentation pipeline..."
echo "=========================================="

# Run tests
python test_augmentations.py

echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Check if config file exists
if [ ! -f "config.yaml" ]; then
    echo "Warning: config.yaml not found. Creating default config..."
    cat > config.yaml << EOL
# VQGAN Pretraining Configuration
input_channels: 3
output_channels: 1
image_size: 512
hidden_dim: 256
num_residual_layers: 2
batch_size: 8
learning_rate: 0.0001
num_epochs: 100
device: "cuda"
input_path: "/blue/jie.xu/pengy1/AR_RM_backup/ICASSP2025_Dataset/Inputs/Task_1_ICASSP"
output_path: "/blue/jie.xu/pengy1/AR_RM_backup/ICASSP2025_Dataset/Outputs/Task_1_ICASSP"
checkpoint_dir: "./checkpoints"
log_dir: "./logs"
results_dir: "./results"
augmentation_preset: "vqgan"
train_split: 0.8
val_split: 0.1
test_split: 0.1
num_workers: 4
save_every: 10
validate_every: 5
buildings: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
antennas: [1]
frequencies: [1]
samples_per_config: 50
EOL
fi

# Check if data paths exist
INPUT_PATH="/blue/jie.xu/pengy1/AR_RM_backup/ICASSP2025_Dataset/Inputs/Task_1_ICASSP"
OUTPUT_PATH="/blue/jie.xu/pengy1/AR_RM_backup/ICASSP2025_Dataset/Outputs/Task_1_ICASSP"

if [ ! -d "$INPUT_PATH" ]; then
    echo "Warning: Input path does not exist: $INPUT_PATH"
    echo "Please update the config.yaml file with the correct paths"
    exit 1
fi

if [ ! -d "$OUTPUT_PATH" ]; then
    echo "Warning: Output path does not exist: $OUTPUT_PATH"
    echo "Please update the config.yaml file with the correct paths"
    exit 1
fi

# Run training
echo "Starting VQGAN pretraining..."
python train_vqgan_pretraining.py --config config.yaml

echo "=========================================="
echo "Training completed!"
echo "=========================================="

echo "Results saved in:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Logs: ./logs/"
echo "  - Results: ./results/"

echo ""
echo "To resume training from a checkpoint:"
echo "  python train_vqgan_pretraining.py --config config.yaml --resume ./checkpoints/checkpoint_epoch_X.pth"

echo ""
echo "To modify training parameters, edit config.yaml and restart training."