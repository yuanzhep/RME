#!/bin/bash

echo "FROZEN CODEBOOK VQGAN PRETRAINING"
echo "===================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# CRITICAL: Path to existing VQGAN weights (for codebook extraction)
PRETRAINED_CODEBOOK_PATH="models/pytorch_model.bin"

# Paths
SCENE_DIR="/blue/jie.xu/pengy1/AR_RM_backup/ripple/dataset/map"
PATHLOSS_DIR="/blue/jie.xu/pengy1/AR_RM_backup/ripple/dataset/pathloss"

# Training parameters 
EPOCHS=65           # "65 epochs"
BATCH_SIZE=16        # "batch size of 16"
LEARNING_RATE=1e-4   # "learning rate of 1×10^-4"
COMMITMENT_WEIGHT=0.25  "λ = 0.25"

# Model parameters
EMBED_DIM=256
N_EMBED=8192
CHANNELS=3
IMAGE_SIZE=256

# Output directory
CHECKPOINT_DIR="./checkpoints_frozen_codebook"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

echo "Configuration:"
echo "  Scene maps: $SCENE_DIR"
echo "  Pathloss maps: $PATHLOSS_DIR"
echo "  Pretrained codebook: $PRETRAINED_CODEBOOK_PATH"
echo "  Training: $EPOCHS epochs, batch=$BATCH_SIZE, lr=$LEARNING_RATE"
echo "  Commitment weight λ = $COMMITMENT_WEIGHT"
echo ""

# Check paths exist
if [ ! -d "$SCENE_DIR" ]; then
    echo "Scene directory not found: $SCENE_DIR"
    exit 1
fi

if [ ! -d "$PATHLOSS_DIR" ]; then
    echo "Pathloss directory not found: $PATHLOSS_DIR"
    exit 1
fi

if [ ! -f "$PRETRAINED_CODEBOOK_PATH" ]; then
    echo "Pretrained codebook not found: $PRETRAINED_CODEBOOK_PATH"
    echo "Please provide path to your existing VQGAN weights"
    exit 1
fi

echo "All paths verified"
echo ""
echo "Starting training..."

# Run training
python frozen_codebook_pretrain.py \
    --scene_dir $SCENE_DIR \
    --pathloss_dir $PATHLOSS_DIR \
    --pretrained_codebook_path $PRETRAINED_CODEBOOK_PATH \
    --embed_dim $EMBED_DIM \
    --n_embed $N_EMBED \
    --ch $CHANNELS \
    --image_size $IMAGE_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --commitment_weight $COMMITMENT_WEIGHT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --num_workers 4 \
    2>&1 | tee $CHECKPOINT_DIR/training.log

echo ""
echo "Training completed!"
echo "Results saved in: $CHECKPOINT_DIR"
echo "Training log: $CHECKPOINT_DIR/training.log"
echo ""
echo "Key benefits achieved:"
echo "  Codebook frozen - LLaMA compatibility maintained"
echo "  Encoder optimized for 3-channel scene tensors"
echo "  Decoder specialized for pathloss reconstruction"