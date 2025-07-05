#!/usr/bin/env python3
"""
Simple Live Chat Demo with GNN-based Context Retrieval
Clean implementation: add message → score → retrieve → send to LLM
"""

import argparse
import torch
from openai import OpenAI
import logging
import sys
from pathlib import Path

from train_gnn import ConversationGNN
from simple_graph_manager import SimpleGraphManager

# Import configuration
try:
    from config import DEFAULT_MODEL_PATH
except ImportError:
    DEFAULT_MODEL_PATH = Path("checkpoints/models/conversation_gnn.pt")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose output from transformers/sentence_transformers during chat
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)


def run_chat_loop(graph_manager: SimpleGraphManager,
                 openai_client: OpenAI,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 1.5,
                 context_window: int = 10,
                 min_recent: int = 3):
    """
    Simple chat loop: user input → graph → context retrieval → LLM → graph
    """
    # Temperature presets for different query types
    temperature_presets = {
        'follow_up': 0.8,      # Focus on recent messages
        'clarification': 1.0,   # Balanced
        'reference': 1.5,       # Look moderately far back
        'debugging': 2.0,       # Look far back for error context
        'summary': 2.5,         # Very broad context
        'exploration': 3.0      # Explore full conversation
    }
    
    print("\n" + "="*60)
    print("SIMPLE LIVE CHAT WITH GNN CONTEXT RETRIEVAL")
    print(f"Temperature: {temperature}, Context window: {context_window}")
    print("="*60)
    print("\nCommands:")
    print("  'quit' - exit")
    print("  'stats' - show graph statistics")
    print("  'temp <value>' - change temperature")
    print("  'temp presets' - show temperature presets")
    print("\n")

    system_prompt = "You are a helpful assistant. Use the conversation context to provide relevant responses."

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stats':
            stats = graph_manager.get_graph_stats()
            print(f"\nGraph stats: {stats}")
            continue
        elif user_input.lower().startswith('temp '):
            parts = user_input.split()
            if len(parts) > 1:
                if parts[1] == 'presets':
                    print("\nTemperature presets:")
                    for name, value in temperature_presets.items():
                        print(f"  {name}: {value}")
                    continue
                else:
                    try:
                        temperature = float(parts[1])
                        print(f"Temperature set to {temperature}")
                    except:
                        print("Usage: temp <float> | temp presets")
            continue

        # Add user message to graph
        user_idx = graph_manager.add_message('user', user_input)

        # Build messages for LLM (context + user message)
        messages = graph_manager.build_llm_messages(
            user_idx,
            system_prompt,
            temperature=temperature,
            context_window=context_window,
            min_recent=min_recent
        )

        # Show context info
        context_count = len(messages) - 2  # Minus system and user message
        print(f"\n[Using {context_count} context messages]")

        try:
            # Get LLM response
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
            )

            assistant_response = response.choices[0].message.content
            print(f"\nAssistant: {assistant_response}")

            # Add assistant response to graph
            graph_manager.add_message('assistant', assistant_response)

        except Exception as e:
            print(f"\nError calling OpenAI: {e}")
            print("Continuing...")


def load_conversation_history(graph_manager: SimpleGraphManager, history_file: str):
    """Load conversation history from JSON file"""
    import json

    with open(history_file, 'r') as f:
        data = json.load(f)

    print(f"\nLoading conversation history from {history_file}...")

    # Check if it's our scripted format
    if 'turns' in data:
        # Scripted format
        for turn in data['turns']:
            graph_manager.add_message('user', turn['user'])
            if 'assistant' in turn:
                graph_manager.add_message('assistant', turn['assistant'])
    else:
        # Simple list format
        for msg in data:
            graph_manager.add_message(msg['role'], msg['text'])

    stats = graph_manager.get_graph_stats()
    print(f"Loaded {stats['num_messages']} messages")
    print(f"Graph has {stats['num_edges']} edges ({stats['temporal_edges']} temporal, {stats['semantic_edges']} semantic)")


def main():
    parser = argparse.ArgumentParser(description='Simple live chat with GNN context retrieval')
    parser.add_argument('--gnn-checkpoint', type=str, default=str(DEFAULT_MODEL_PATH),
                        help='Path to trained GNN model')
    parser.add_argument('--openai-key', type=str, required=True,
                        help='OpenAI API key')
    parser.add_argument('--openai-base-url', type=str, default="https://api.openai.com/v1",
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo",
                        help='OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Temperature for GNN attention')
    parser.add_argument('--context-window', type=int, default=10,
                        help='Maximum context messages to retrieve')
    parser.add_argument('--min-recent', type=int, default=3,
                        help='Minimum recent messages to always include')
    parser.add_argument('--semantic-threshold', type=float, default=0.75,
                        help='Similarity threshold for semantic edges')
    parser.add_argument('--load-history', type=str,
                        help='Load conversation history from JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load GNN model
    logger.info(f"Loading GNN model from {args.gnn_checkpoint}")
    checkpoint = torch.load(args.gnn_checkpoint, map_location=args.device)

    model = ConversationGNN(
        input_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['args']['hidden_dim'],
        output_dim=checkpoint['args']['output_dim'],
        num_heads=checkpoint['args']['num_heads'],
        dropout=checkpoint['args']['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create graph manager
    graph_manager = SimpleGraphManager(
        gnn_model=model,
        semantic_threshold=args.semantic_threshold,
        device=args.device
    )

    # Load history if provided
    if args.load_history:
        load_conversation_history(graph_manager, args.load_history)

    # Create OpenAI client
    openai_client = OpenAI(
        api_key=args.openai_key,
        base_url=args.openai_base_url
    )

    # Run chat loop
    run_chat_loop(
        graph_manager,
        openai_client,
        args.model,
        temperature=args.temperature,
        context_window=args.context_window,
        min_recent=args.min_recent
    )


if __name__ == "__main__":
    main()
