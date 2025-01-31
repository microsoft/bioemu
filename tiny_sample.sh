#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


# Generate samples for a tiny sequence.

CHECKPOINT_DIR=$SCRIPT_DIR/checkpoints/bioemu-v1.0
MODEL_CONFIG_PATH=$CHECKPOINT_DIR/config.yaml
CKPT_PATH=$CHECKPOINT_DIR/checkpoint.ckpt
OUTPUT_DIR=$HOME/bioemu-output
DENOISER_CONFIG_PATH=$SCRIPT_DIR/configs/denoiser/dpm.yaml 

SEQUENCE="GYDPETGTWG"
python -m bioemu.sample --ckpt_path $CKPT_PATH \
                   --model_config_path $MODEL_CONFIG_PATH \
                   --denoiser_config_path $DENOISER_CONFIG_PATH \
                   --sequence $SEQUENCE \
                   --num_samples 100 \
                   --batch_size_100 10 \
                   --output_dir $OUTPUT_DIR