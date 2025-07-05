#!/usr/bin/env python3
"""
Unified Evaluation Script for Conversation GNN
Supports both offline evaluation on test data and live REPL mode
Uses PyTorch Geometric graph structure for consistency
"""

import argparse
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from openai import OpenAI

from train_gnn import ConversationGNN, load_conversations
from simple_graph_manager import SimpleGraphManager

# Import configuration
try:
    from config import DEFAULT_MODEL_PATH, RAW_DATA_DIR, OUTPUT_DIR
except ImportError:
    DEFAULT_MODEL_PATH = Path("checkpoints/models/conversation_gnn.pt")
    RAW_DATA_DIR = Path("data/raw")
    OUTPUT_DIR = Path("outputs")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationEvaluator:
    """Unified evaluator for the Conversation GNN"""
    
    def __init__(self, model_path: str, device: torch.device):
        """Initialize evaluator with trained model"""
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model with saved parameters
        self.embedding_dim = checkpoint.get('embedding_dim', 384)
        args = checkpoint.get('args', {})
        
        self.model = ConversationGNN(
            input_dim=self.embedding_dim,
            hidden_dim=args.get('hidden_dim', 256),
            output_dim=args.get('output_dim', 128),
            num_heads=args.get('num_heads', 4),
            dropout=args.get('dropout', 0.1)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        self.device = device
        
        logger.info(f"Model loaded. Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}, "
                   f"Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    def evaluate_on_conversation(self, conversation: List[Dict], 
                               temperature: float = 1.5,
                               visualize: bool = False,
                               save_path: Optional[str] = None) -> Dict:
        """
        Evaluate model performance on a single conversation
        
        Returns:
            Dictionary with evaluation metrics and analysis
        """
        # Create graph manager for this conversation
        graph_manager = SimpleGraphManager(
            gnn_model=self.model,
            semantic_threshold=0.75,
            device=self.device
        )
        
        # Build conversation graph
        for msg in conversation:
            graph_manager.add_message(msg['role'], msg['text'])
        
        # Analyze attention patterns
        results = {
            'num_messages': len(conversation),
            'graph_stats': graph_manager.get_graph_stats(),
            'attention_analysis': []
        }
        
        # Analyze attention for each message
        for i in range(1, len(conversation)):
            context, indices = graph_manager.get_relevant_context(
                query_idx=i,
                temperature=temperature,
                top_k=min(10, i),
                min_recent=min(3, i)
            )
            
            # Analyze if the model captures dependencies correctly
            msg = conversation[i]
            if msg.get('is_context_dependent', False) and msg.get('depends_on_indices'):
                # Check if model found the labeled dependencies
                found_deps = set(indices) & set(msg['depends_on_indices'])
                precision = len(found_deps) / len(indices) if indices else 0
                recall = len(found_deps) / len(msg['depends_on_indices']) if msg['depends_on_indices'] else 0
                
                results['attention_analysis'].append({
                    'message_idx': i,
                    'true_dependencies': msg['depends_on_indices'],
                    'predicted_context': indices,
                    'precision': precision,
                    'recall': recall,
                    'dependency_type': msg.get('dependency_type')
                })
        
        if visualize and save_path:
            self._visualize_attention_pattern(conversation, graph_manager, temperature, save_path)
        
        return results
    
    def _visualize_attention_pattern(self, conversation: List[Dict], 
                                   graph_manager: SimpleGraphManager,
                                   temperature: float,
                                   save_path: str):
        """Visualize attention patterns for a conversation"""
        n = len(conversation)
        attention_matrix = np.zeros((n, n))
        
        # Compute attention scores for all pairs
        embeddings = graph_manager.get_gnn_embeddings()
        
        with torch.no_grad():
            for i in range(n):
                for j in range(i):
                    score = self.model.score_relevance(
                        embeddings[i].unsqueeze(0),
                        embeddings[j].unsqueeze(0),
                        temperature=temperature
                    ).item()
                    attention_matrix[i, j] = score
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create labels showing role and first few words
        labels = []
        for i, msg in enumerate(conversation):
            text_preview = msg['text'][:30] + "..." if len(msg['text']) > 30 else msg['text']
            labels.append(f"{i}: {msg['role'][:4]} - {text_preview}")
        
        # Plot heatmap
        sns.heatmap(attention_matrix, 
                   cmap='YlOrRd', 
                   cbar=True, 
                   ax=ax,
                   xticklabels=range(n),
                   yticklabels=labels)
        
        ax.set_title(f'Attention Pattern (Temperature={temperature})')
        ax.set_xlabel('Context Message Index')
        ax.set_ylabel('Query Message')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention visualization to {save_path}")
    
    def evaluate_dataset(self, conversations: List[List[Dict]], 
                        num_samples: int = 10,
                        temperature: float = 1.5) -> Dict:
        """Evaluate on multiple conversations and compute aggregate metrics"""
        logger.info(f"Evaluating on {min(num_samples, len(conversations))} conversations")
        
        all_results = []
        precision_scores = []
        recall_scores = []
        
        output_dir = OUTPUT_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, conv_data in enumerate(conversations[:num_samples]):
            messages = conv_data["messages"]
            if len(messages) < 5:  # Skip very short conversations
                continue
                
            # Evaluate conversation
            results = self.evaluate_on_conversation(
                messages, 
                temperature=temperature,
                visualize=True,
                save_path=str(output_dir / f"attention_conv_{i}.png")
            )
            
            all_results.append(results)
            
            # Collect precision/recall scores
            for analysis in results['attention_analysis']:
                if 'precision' in analysis:
                    precision_scores.append(analysis['precision'])
                    recall_scores.append(analysis['recall'])
        
        # Compute aggregate metrics
        aggregate_metrics = {
            'num_conversations': len(all_results),
            'avg_precision': np.mean(precision_scores) if precision_scores else 0,
            'avg_recall': np.mean(recall_scores) if recall_scores else 0,
            'temperature_used': temperature,
            'output_directory': str(output_dir)
        }
        
        # Save detailed results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump({
                'aggregate_metrics': aggregate_metrics,
                'detailed_results': all_results
            }, f, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
        
        return aggregate_metrics


def run_live_repl(graph_manager: SimpleGraphManager, 
                 openai_client: Optional[OpenAI] = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 1.5):
    """
    Run interactive REPL mode for testing the model
    """
    print("\n" + "="*60)
    print("CONVERSATION GNN - INTERACTIVE REPL MODE")
    print("="*60)
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit REPL")
    print("  'stats' - Show graph statistics")
    print("  'temp <value>' - Change temperature (current: {:.1f})".format(temperature))
    print("  'context' - Show retrieved context for last message")
    print("  'clear' - Clear conversation and start fresh")
    print("\n")
    
    if openai_client:
        print("OpenAI integration enabled. Assistant will respond to your messages.")
    else:
        print("OpenAI integration disabled. Only context retrieval will be shown.")
    
    print("-" * 60)
    
    system_prompt = "You are a helpful assistant. Use the conversation context to provide relevant responses."
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
                
            elif user_input.lower() == 'stats':
                stats = graph_manager.get_graph_stats()
                print(f"\nGraph Statistics:")
                print(f"  Messages: {stats['num_messages']}")
                print(f"  Total edges: {stats['num_edges']}")
                print(f"  Temporal edges: {stats['temporal_edges']}")
                print(f"  Semantic edges: {stats['semantic_edges']}")
                continue
                
            elif user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: temp <float value>")
                continue
                
            elif user_input.lower() == 'context':
                if graph_manager.messages:
                    last_idx = len(graph_manager.messages) - 1
                    context, indices = graph_manager.get_relevant_context(
                        last_idx, temperature=temperature
                    )
                    print(f"\nContext for last message (indices: {indices}):")
                    for i, (msg, idx) in enumerate(zip(context, indices)):
                        print(f"  [{idx}] {msg['role']}: {msg['text'][:100]}...")
                else:
                    print("No messages in conversation yet.")
                continue
                
            elif user_input.lower() == 'clear':
                # Reinitialize graph manager
                graph_manager = SimpleGraphManager(
                    gnn_model=graph_manager.gnn_model,
                    semantic_threshold=graph_manager.semantic_threshold,
                    device=graph_manager.device
                )
                print("Conversation cleared.")
                continue
            
            # Add user message
            user_idx = graph_manager.add_message('user', user_input)
            
            if openai_client:
                # Get context and generate response
                messages = graph_manager.build_llm_messages(
                    user_idx,
                    system_prompt,
                    temperature=temperature,
                    context_window=10,
                    min_recent=3
                )
                
                # Show context info
                context_count = len(messages) - 2  # Minus system and user
                print(f"\n[Retrieved {context_count} context messages with temperature={temperature}]")
                
                try:
                    # Get LLM response
                    response = openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=200
                    )
                    
                    assistant_response = response.choices[0].message.content
                    print(f"\nAssistant: {assistant_response}")
                    
                    # Add to graph
                    graph_manager.add_message('assistant', assistant_response)
                    
                except Exception as e:
                    print(f"\nError calling OpenAI: {e}")
            else:
                # Just show what context would be retrieved
                context, indices = graph_manager.get_relevant_context(
                    user_idx,
                    temperature=temperature,
                    top_k=10,
                    min_recent=3
                )
                print(f"\n[Would retrieve {len(context)} messages as context: indices {indices}]")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation for Conversation GNN')
    
    # Model and data paths
    parser.add_argument('--model-path', type=str, default=str(DEFAULT_MODEL_PATH),
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-path', type=str, 
                        default=str(RAW_DATA_DIR / 'conversations.json'),
                        help='Path to conversations for evaluation')
    
    # Evaluation mode
    parser.add_argument('--mode', choices=['dataset', 'repl', 'both'], default='dataset',
                        help='Evaluation mode: dataset analysis, REPL, or both')
    
    # Evaluation parameters
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of conversations to evaluate (dataset mode)')
    parser.add_argument('--temperature', type=float, default=1.5,
                        help='Temperature for attention computation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/mps/cpu/auto)')
    
    # REPL mode options
    parser.add_argument('--openai-key', type=str,
                        help='OpenAI API key for REPL mode with LLM')
    parser.add_argument('--openai-base-url', type=str, default="https://api.openai.com/v1",
                        help='OpenAI API base URL')
    parser.add_argument('--llm-model', type=str, default="gpt-3.5-turbo",
                        help='LLM model to use in REPL mode')
    parser.add_argument('--load-conversation', type=str,
                        help='Load a specific conversation index for REPL mode')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(args.model_path, device)
    
    if args.mode in ['dataset', 'both']:
        # Load conversations
        conversations = load_conversations([args.data_path])
        logger.info(f"Loaded {len(conversations)} conversations")
        
        # Run dataset evaluation
        metrics = evaluator.evaluate_dataset(
            conversations,
            num_samples=args.num_samples,
            temperature=args.temperature
        )
        
        print("\n" + "="*60)
        print("DATASET EVALUATION RESULTS")
        print("="*60)
        print(f"Evaluated {metrics['num_conversations']} conversations")
        print(f"Average Precision: {metrics['avg_precision']:.3f}")
        print(f"Average Recall: {metrics['avg_recall']:.3f}")
        print(f"Results saved to: {metrics['output_directory']}")
        print("="*60 + "\n")
    
    if args.mode in ['repl', 'both']:
        # Create graph manager for REPL
        graph_manager = SimpleGraphManager(
            gnn_model=evaluator.model,
            semantic_threshold=0.75,
            device=device
        )
        
        # Load conversation if specified
        if args.load_conversation:
            conversations = load_conversations([args.data_path])
            conv_idx = int(args.load_conversation)
            if 0 <= conv_idx < len(conversations):
                print(f"\nLoading conversation {conv_idx} into graph...")
                for msg in conversations[conv_idx]:
                    graph_manager.add_message(msg['role'], msg['text'])
                print(f"Loaded {len(conversations[conv_idx])} messages")
            else:
                print(f"Warning: Conversation index {conv_idx} out of range")
        
        # Create OpenAI client if key provided
        openai_client = None
        if args.openai_key:
            openai_client = OpenAI(
                api_key=args.openai_key,
                base_url=args.openai_base_url
            )
        
        # Run REPL
        run_live_repl(
            graph_manager,
            openai_client,
            args.llm_model,
            args.temperature
        )


if __name__ == "__main__":
    main()