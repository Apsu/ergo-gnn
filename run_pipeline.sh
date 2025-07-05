#!/bin/bash
# Example pipeline script for the GNN conversation system

set -e  # Exit on error

echo "🚀 GNN Conversation System Training Pipeline"
echo "==========================================="

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

# Default values
MODEL=${MODEL:-"gpt-4.1"}
API_URL=${API_URL:-"https://api.openai.com/v1"}
CONVERSATION_COUNT=${CONVERSATION_COUNT:-100}

echo "📁 Directory Structure:"
echo "  - Data: data/raw/, data/processed/"
echo "  - Models: checkpoints/models/"
echo "  - Training: checkpoints/training/"
echo "  - Logs: logs/"
echo ""

# Step 1: Generate conversations
echo "📝 Step 1: Generating conversations..."
if [ -f "data/raw/conversations.json" ] && [ "$1" != "--force" ]; then
    echo "  ℹ️  Conversations already exist. Use --force to regenerate or --resume to continue."
    if [ "$1" == "--resume" ]; then
        python generate_gnn_data.py \
            --api-key "$OPENAI_API_KEY" \
            --base-url "$API_URL" \
            --model "$MODEL" \
            --count "$CONVERSATION_COUNT" \
            --resume
    else
        echo "  ⏭️  Skipping generation step."
    fi
else
    python generate_gnn_data.py \
        --api-key "$OPENAI_API_KEY" \
        --base-url "$API_URL" \
        --model "$MODEL" \
        --count "$CONVERSATION_COUNT"
fi

# Step 2: Train the model
echo ""
echo "🧠 Step 2: Training the GNN model..."
if [ -f "checkpoints/models/conversation_gnn.pt" ] && [ "$1" != "--force" ]; then
    echo "  ℹ️  Model already exists. Use --force to retrain."
    echo "  ⏭️  Skipping training step."
else
    # Train using the generated conversations from data/raw/
    python train_gnn.py --data-paths data/raw/conversations.json
fi

# Step 3: Summary
echo ""
echo "✅ Pipeline Complete!"
echo ""
echo "📊 Results:"
if [ -f "data/raw/conversations.json" ]; then
    echo "  - Conversations: $(jq length data/raw/conversations.json) generated"
fi
if [ -f "checkpoints/models/conversation_gnn.pt" ]; then
    echo "  - Model saved: checkpoints/models/conversation_gnn.pt"
fi

echo ""
echo "🔍 Next steps:"
echo "  - Evaluate the model: python evaluate_unified.py --mode dataset"
echo "  - Interactive REPL: python evaluate_unified.py --mode repl"
echo "  - Live chat demo: python simple_live_chat.py --openai-key YOUR_KEY"
