#!/bin/bash
# Run two staggered easy mode instances for better GPU saturation
# Instance A starts immediately, Instance B starts after delay

DELAY=${1:-14}  # Default 14s delay (full cycle offset for interleaving)

echo "Starting dual easy mode training with ${DELAY}s stagger..."

# Instance A - Bella
python train_easy.py --preset large --epochs 100 --bella --steps-per-epoch 400 --batch-size 1024 --no-latent-cache \
    --output-dir jax_ckpt_easy_A --name dual_A 2>&1 | sed 's/^/[A] /' &
PID_A=$!

# Wait for stagger
echo "Waiting ${DELAY}s before starting instance B..."
sleep $DELAY

# Instance B - Synthetic levels (different data source for diversity)
python train_easy.py --preset large --epochs 100 --levels 0-7 --steps-per-epoch 400 --batch-size 1024 --no-latent-cache \
    --output-dir jax_ckpt_easy_B --name dual_B 2>&1 | sed 's/^/[B] /' &
PID_B=$!

echo "Both instances running: A=$PID_A, B=$PID_B"
echo "Press Ctrl+C to stop both"

# Wait for both
wait $PID_A $PID_B
