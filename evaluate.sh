#!/bin/bash
# Evaluation script for the GNN conversation system
#
# Usage:
#   ./evaluate.sh                           # Run with defaults
#   NUM_SAMPLES=5 ./evaluate.sh            # Evaluate 5 samples
#   MODE=repl ./evaluate.sh                # Run in REPL mode
#   DEVICE=mps ./evaluate.sh               # Use MPS device

set -e  # Exit on error

echo "🔍 GNN Conversation System Evaluation"
echo "===================================="

# Default values
MODE=${MODE:-"dataset"}
NUM_SAMPLES=${NUM_SAMPLES:-10}
DEVICE=${DEVICE:-"auto"}
MODEL_PATH=${MODEL_PATH:-"checkpoints/models/conversation_gnn.pt"}
DATA_PATH=${DATA_PATH:-"data/raw/conversations.json"}
TEMPERATURE=${TEMPERATURE:-1.5}

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    echo "   Please train a model first using: ./run_pipeline.sh"
    exit 1
fi

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data not found at $DATA_PATH"
    echo "   Please generate data first using: ./run_pipeline.sh"
    exit 1
fi

echo "📊 Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Data: $DATA_PATH"
echo "  - Mode: $MODE"
echo "  - Device: $DEVICE"
echo "  - Samples: $NUM_SAMPLES"
echo "  - Temperature: $TEMPERATURE"
echo ""

# Run evaluation
echo "🚀 Running evaluation..."
python evaluate_unified.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --mode "$MODE" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --device "$DEVICE" \
    "$@"  # Pass any additional arguments

echo ""
echo "✅ Evaluation complete!"