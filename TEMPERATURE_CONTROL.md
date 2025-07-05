# Temperature Control for GNN-based Conversation Retrieval

## Overview

The temperature parameter controls how the GNN distributes attention across conversation history:
- **Low temperature (0.5-1.0)**: Focus on recent messages
- **Medium temperature (1.5-2.0)**: Balanced attention
- **High temperature (2.5-3.0+)**: Look broadly across entire history

## Key Files

### 1. `train_gnn.py`
- `score_relevance()` method accepts temperature parameter
- Default temperature = 1.0 during training
- No retraining needed for different temperatures at inference

### 2. `simple_graph_manager.py`
- Core graph management with temperature support
- `get_relevant_context()` and `build_llm_messages()` accept temperature parameter
- Default temperature = 1.5 for balanced retrieval

### 3. `evaluate_unified.py`
- `--temperature` flag for evaluation
- Supports both dataset evaluation and REPL modes
- Generates attention visualization heatmaps

### 4. `simple_live_chat.py`
- Live chat demo with temperature control
- `temp <value>` command to adjust temperature dynamically
- `temp presets` shows recommended values:
  - `follow_up`: 0.8 (recent focus)
  - `clarification`: 1.0 (balanced)
  - `reference`: 1.5 (look moderately far back)
  - `debugging`: 2.0 (look far back)
  - `summary`: 2.5 (very broad context)
  - `exploration`: 3.0 (full history)

## Usage Examples

### Dataset Evaluation with Temperature
```bash
python evaluate_unified.py \
  --model-path checkpoints/models/conversation_gnn.pt \
  --mode dataset \
  --temperature 2.0 \
  --num-samples 10
```

### Interactive REPL Mode
```bash
python evaluate_unified.py \
  --model-path checkpoints/models/conversation_gnn.pt \
  --mode repl \
  --openai-key YOUR_KEY \
  --temperature 1.5
```

### Live Chat Demo
```bash
python simple_live_chat.py \
  --gnn-checkpoint checkpoints/models/conversation_gnn.pt \
  --openai-key YOUR_KEY \
  --temperature 1.5
```

### Interactive Commands
In both `simple_live_chat.py` and REPL mode:
- `temp 2.0` - Change temperature dynamically
- `temp presets` - Show temperature presets
- `stats` - Show graph statistics
- `quit` - Exit

## Temperature Effects

### Low Temperature (0.5-0.8)
- Strong recency bias
- Good for follow-up questions
- Attention concentrated on last 2-3 messages

### Medium Temperature (1.0-1.5)
- Balanced attention decay
- Good for general conversation
- Considers ~5-8 relevant messages

### High Temperature (2.0-3.0)
- Broad attention distribution
- Good for debugging or summary tasks
- Can retrieve context from 10+ messages back

## How It Works

1. **Scoring**: `score = attention_scorer(combined_embeddings) / temperature`
2. **Softmax**: Higher temperature → flatter distribution
3. **Retrieval**: Top-k messages based on attention scores
4. **Graph Traversal**: From high-scoring nodes, traverse temporal/semantic edges

## Best Practices

1. **Start with default** (1.5) and adjust based on retrieval quality
2. **Use presets** for known query types
3. **Monitor attention span** - if too narrow/broad, adjust temperature
4. **Test with temperature sweep** to find optimal values for your use case

## API Example

```python
from simple_graph_manager import SimpleGraphManager
from train_gnn import ConversationGNN

# Load model
model = ConversationGNN(...)
manager = SimpleGraphManager(model)

# Add messages
manager.add_message('user', 'Tell me about caching strategies')
manager.add_message('assistant', 'Caching is a technique...')

# Get context with custom temperature
context, indices = manager.get_relevant_context(
    query_idx=1,
    temperature=2.0,  # Look broadly for related context
    top_k=10
)

# Build LLM messages with temperature
messages = manager.build_llm_messages(
    query_idx=1,
    system_prompt="You are a helpful assistant",
    temperature=1.5  # Balanced retrieval
)
```