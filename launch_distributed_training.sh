#!/bin/bash
# Launch distributed training with torchrun (replacement for torch.distributed.launch)

# Example usage:
# Single node, multiple GPUs:
# ./launch_distributed_training.sh

# Multiple nodes (example for 2 nodes):
# On node 0: ./launch_distributed_training.sh 0 2
# On node 1: ./launch_distributed_training.sh 1 2

# Default values
NODE_RANK=${1:-0}
NNODES=${2:-1}
NPROC_PER_NODE=${3:-$(nvidia-smi -L | wc -l)}  # Auto-detect number of GPUs
MASTER_ADDR=${4:-127.0.0.1}
MASTER_PORT=${5:-29500}

echo "Launching distributed training:"
echo "  Node rank: $NODE_RANK"
echo "  Number of nodes: $NNODES"
echo "  Processes per node: $NPROC_PER_NODE"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"

# Launch with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_gnn.py \
    --data-paths conversations.json \
    --model-save-path conversation_gnn_distributed.pt \
    --batch-size 16 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --accumulation-steps 2

# Note: The --distributed flag is no longer needed as the script auto-detects
# distributed mode based on environment variables set by torchrun
