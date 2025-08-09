#!/bin/bash

echo "VQGAN PRETRAINING"
echo "===================================="
export CUDA_VISIBLE_DEVICES=0
PRETRAINED_CODEBOOK_PATH=".../models/pytorch_model.bin"
SCENE_DIR=".../ripple/dataset/..."
PATHLOSS_DIR=".../ripple/dataset/..."
EPOCHS=80         
BATCH_SIZE=256      
LEARNING_RATE=1e-4   
COMMITMENT_WEIGHT=0.25  
EMBED_DIM=256
N_EMBED=8192
CHANNELS=3
IMAGE_SIZE=256
CHECKPOINT_DIR="./checkpoints"

mkdir -p $CHECKPOINT_DIR
echo "Configuration:"
echo "  Scene maps: $SCENE_DIR"
echo "  Pathloss maps: $PATHLOSS_DIR"
echo "  Pretrained codebook: $PRETRAINED_CODEBOOK_PATH"
echo "  Training: $EPOCHS epochs, batch=$BATCH_SIZE, lr=$LEARNING_RATE"
echo "  Commitment weight λ = $COMMITMENT_WEIGHT"

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

python frozen.py \
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
echo "completed"
echo "Results saved in: $CHECKPOINT_DIR"
echo "Training log: $CHECKPOINT_DIR/training.log"
echo ""
