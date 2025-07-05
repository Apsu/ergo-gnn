# GNN-Based Conversation Context Retrieval

A Graph Neural Network system for intelligent conversation context retrieval that maintains unlimited conversation history while efficiently retrieving relevant messages.

## Quick Start

### Installation
```bash
pip install torch torch-geometric transformers sentence-transformers openai
# or with uv:
uv pip install torch torch-geometric transformers sentence-transformers openai
```

### Automated Pipeline
```bash
# Run complete pipeline (generates data and trains model)
./run_pipeline.sh

# Resume interrupted generation
./run_pipeline.sh --resume

# Force regenerate everything
./run_pipeline.sh --force
```

### Manual Steps
```bash
# 1. Generate training data (saves to data/raw/)
python generate_gnn_data.py \
    --api-key YOUR_KEY \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-2024-08-06 \
    --count 1000

# 2. Train GNN (loads from data/raw/, saves to checkpoints/models/)
python train_gnn.py

# 3. Live chat demo (loads from checkpoints/models/)
python simple_live_chat.py --openai-key YOUR_KEY
```

### Directory Structure
```
ergo-gnn/
├── data/
│   ├── raw/          # Generated conversations
│   └── processed/    # Processed data
├── checkpoints/
│   ├── models/       # Trained models (conversation_gnn.pt)
│   └── training/     # Training checkpoints (epoch_*.pt)
├── logs/             # Training logs
└── outputs/          # Evaluation results
```

## Key Features

- **Unlimited Context**: No fixed context window limitations
- **Intelligent Retrieval**: GNN learns what's relevant, not just recent
- **Graph-Based**: Messages connected by temporal and semantic edges  
- **Temperature Control**: Adjust attention focus without retraining
- **Production Ready**: Efficient caching and incremental updates

## Core Components

### SimpleGraphManager
Maintains the conversation graph with automatic edge creation:
```python
manager = SimpleGraphManager(gnn_model)
msg_id = manager.add_message('user', 'Hello!')
context = manager.get_relevant_context(msg_id)
```

### Temperature Control
Adjust retrieval behavior at inference time:
- `0.5-1.0`: Focus on recent messages
- `1.5`: Balanced (default)
- `2.0-3.0`: Look broadly across history

## How It Works

1. **Messages become nodes** in a graph with embeddings
2. **Edges connect messages** (temporal + semantic)
3. **GNN processes the graph** to learn relevance patterns
4. **Retrieval uses GNN scores** to find context
5. **No heuristics** - the model learns what matters

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Files

- `train_gnn.py` - GNN training script
- `simple_graph_manager.py` - Core graph management with PyTorch Geometric
- `simple_live_chat.py` - Live chat demo
- `generate_gnn_data.py` - Training data generation with LLM labeling
- `evaluate_unified.py` - Unified evaluation script (dataset & REPL modes)

## Evaluation

```bash
# Evaluate on dataset
python evaluate_unified.py --mode dataset --num-samples 20

# Interactive REPL mode
python evaluate_unified.py --mode repl --openai-key YOUR_KEY

# Load specific conversation into REPL
python evaluate_unified.py --mode repl --load-conversation 5
```

## Example Usage

```python
from simple_graph_manager import SimpleGraphManager
from train_gnn import ConversationGNN

# Load model
model = ConversationGNN(...)
manager = SimpleGraphManager(model)

# Build conversation
manager.add_message('user', 'Tell me about Redis')
manager.add_message('assistant', 'Redis is a fast in-memory...')
manager.add_message('user', 'How do I set it up?')

# Get context for latest message
messages = manager.build_llm_messages(
    query_idx=2,
    system_prompt="You are a helpful assistant",
    temperature=1.5
)
```

## Performance

- Add message: ~50ms (includes embedding + edge creation)
- Retrieve context: ~100ms for 1000-message history  
- Memory: ~1GB for 10,000 messages
- GPU provides 3-5x speedup

## Project Status

The project is functional with all core features working:
- ✅ LLM-supervised learning for context dependencies
- ✅ Graph construction with temporal and semantic edges
- ✅ Temperature-controlled attention mechanism
- ✅ Checkpoint/resume for data generation
- ✅ Unified evaluation with dataset and REPL modes

See [TODO.md](TODO.md) for planned improvements and known issues.

## Development

### Running Tests
```bash
# Quick functionality test
NUM_SAMPLES=5 ./evaluate.sh

# Full evaluation
./evaluate.sh
```

### Troubleshooting

**Missing dependencies**: Install with `uv pip install torch torch-geometric transformers sentence-transformers openai matplotlib seaborn`

**MPS errors on Mac**: The scripts auto-detect MPS. Use `--device cpu` to force CPU mode.

**API rate limits**: Use `--resume` flag with generate_gnn_data.py to continue from checkpoints.

## License

MIT