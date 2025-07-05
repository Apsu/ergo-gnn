# Temperature Control for GNN-based Conversation Retrieval

## Overview

The temperature parameter controls how the GNN distributes attention across conversation history:
- **Low temperature (0.5-1.0)**: Focus on recent messages
- **Medium temperature (1.5-2.0)**: Balanced attention
- **High temperature (2.5-3.0+)**: Look broadly across entire history

## Key Files

### 1. `train_gnn.py`
- Modified `score_relevance()` method to accept temperature parameter
- Default temperature = 1.0 during training
- No retraining needed for different temperatures at inference

### 2. `conversation_graph_retriever.py`
- `ConversationGraphRetriever` class for inference-time retrieval
- Temperature presets for different query types:
  - `follow_up`: 0.8 (recent focus)
  - `general`: 1.5 (balanced)
  - `debugging`: 2.0 (look far back)
  - `exploration`: 3.0 (full history)

### 3. `evaluate_conversation_gnn.py`
- `--temperature` flag for evaluation
- `--temperature-sweep` for analyzing effects
- Generates comparison visualizations

### 4. `evaluate_live_chat.py`
- Full pipeline demo with OpenAI integration
- Three modes:
  - `interactive`: Start fresh conversation
  - `scripted`: Run predefined script
  - `repl`: Load history, then interact

## Usage Examples

### Basic Evaluation with Temperature
```bash
python evaluate_conversation_gnn.py \
  --model-path checkpoints/best_model.pt \
  --temperature 2.0
```

### Temperature Sweep Analysis
```bash
python evaluate_conversation_gnn.py \
  --model-path checkpoints/best_model.pt \
  --temperature-sweep
```

### Live Chat Demo (REPL mode)
```bash
python evaluate_live_chat.py \
  --gnn-checkpoint checkpoints/best_model.pt \
  --openai-key YOUR_KEY \
  --demo-mode repl \
  --script-file sample_conversation_script.json \
  --temperature 1.5
```

### Interactive Commands in REPL
- `temp 2.0` - Change temperature dynamically
- `graph` - Visualize conversation graph
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
from conversation_graph_retriever import ConversationGraphRetriever

retriever = ConversationGraphRetriever(model, default_temperature=1.5)

# Manual temperature
context = retriever.get_relevant_context(
    query_embedding, 
    graph_embeddings,
    messages,
    temperature=2.0
)

# Automatic based on query type
context = retriever.get_relevant_context(
    query_embedding,
    graph_embeddings, 
    messages,
    query_type='debugging'  # Uses temperature=2.0
)
```