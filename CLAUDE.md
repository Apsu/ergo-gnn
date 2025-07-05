# Ergo-GNN Project Context

## Project Overview

This is a GNN-based conversation context retrieval system that uses Graph Neural Networks to intelligently select relevant messages from conversation history using LLM-supervised learning.

## Coding Guidelines

1. **No backwards compatibility** - Forward-looking design only
2. **No legacy code or comments** - No references to old versions or migrations
3. **No adapters/converters** - Single clean data format only
4. **No historical comments** - Code should be self-documenting for its current state
5. **Clean interfaces** - Each component has one way to work, no multiple paths

## Architecture

### Core Approach: LLM-Supervised Learning
- The LLM labels conversation dependencies during data generation
- Each message includes:
  - `is_context_dependent`: bool
  - `depends_on_indices`: List[int] 
  - `dependency_type`: Literal enum of dependency types
- The GNN learns from these labels to identify relevant context

### Data Format
Conversations are stored as a list where each conversation contains:
```json
{
  "messages": [
    {
      "role": "user|assistant",
      "text": "message content",
      "is_context_dependent": true,
      "depends_on_indices": [0, 2],
      "dependency_type": "topic_reference"
    }
  ],
  "conversation_patterns": ["topic_shift", "clarification_loop"],
  "config": { /* generation config */ }
}
```

### Key Components
1. **generate_gnn_data.py** - Generates conversations with LLM-provided labels
2. **train_gnn.py** - Trains GNN using supervised learning from labels
3. **simple_graph_manager.py** - PyTorch Geometric graph management
4. **evaluate_unified.py** - Evaluation in dataset and REPL modes
5. **simple_live_chat.py** - Live chat demo with temperature control

### Directory Structure
```
data/
├── raw/                 # Generated conversations with metadata
└── processed/           # Processed data

checkpoints/
├── models/              # Trained models
└── training/            # Training checkpoints

logs/                    # Training logs
outputs/                 # Evaluation outputs
```

## Key Design Decisions

1. **LLM-Supervised Learning**: The model learns from LLM-provided dependency labels rather than heuristics
2. **PyTorch Geometric Only**: All graph operations use PyG, no NetworkX
3. **Single Data Format**: Conversations always include full metadata
4. **Temperature Control**: Dynamic attention adjustment at inference time