#!/usr/bin/env python3
"""
Complete training pipeline for Conversation GNN
Self-supervised learning from conversation structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import random
from tqdm import tqdm
import wandb
from dataclasses import dataclass
import os
import sys
import argparse
import logging
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time

# Import configuration
try:
    from config import (
        RAW_DATA_DIR, MODEL_CHECKPOINT_DIR, TRAINING_CHECKPOINT_DIR,
        DEFAULT_MODEL_PATH, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
        DEFAULT_NUM_EPOCHS, DEFAULT_HIDDEN_DIM, DEFAULT_OUTPUT_DIM,
        DEFAULT_NUM_HEADS, DEFAULT_DROPOUT
    )
except ImportError:
    # Fallback if config.py is not available
    RAW_DATA_DIR = Path("data/raw")
    MODEL_CHECKPOINT_DIR = Path("checkpoints/models")
    TRAINING_CHECKPOINT_DIR = Path("checkpoints/training")
    DEFAULT_MODEL_PATH = MODEL_CHECKPOINT_DIR / "conversation_gnn.pt"
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_NUM_EPOCHS = 50
    DEFAULT_HIDDEN_DIM = 256
    DEFAULT_OUTPUT_DIM = 128
    DEFAULT_NUM_HEADS = 4
    DEFAULT_DROPOUT = 0.1

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    text: str
    role: str
    conversation_id: str
    position: int


class ConversationDataset(Dataset):
    """Dataset for conversation structure learning"""

    def __init__(
        self,
        conversations: List[Dict],
        embeddings: np.ndarray,
        embedding_dim: int,
    ):
        self.conversations = conversations
        self.embedding_dim = embedding_dim
        self.all_messages = []

        # Flatten all messages for indexing
        for conv_id, conv_data in enumerate(conversations):
            messages = conv_data["messages"]
            for pos, msg in enumerate(messages):
                self.all_messages.append(
                    ConversationMessage(
                        text=msg["text"],
                        role=msg["role"],
                        conversation_id=str(conv_id),
                        position=pos,
                    )
                )

        # Use pre-computed embeddings
        self.embeddings = embeddings
        self.embedding_dict = {i: emb for i, emb in enumerate(self.embeddings)}

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv_data = self.conversations[idx]
        messages = conv_data["messages"]

        # Build graph for this conversation
        graph_data = self.build_conversation_graph(messages, idx)

        # Create training targets using dependency labels
        targets = self.create_training_targets(messages, idx, conv_data)

        return graph_data, targets

    def build_conversation_graph(self, messages: List[Dict], conv_idx: int) -> Data:
        """Build graph with temporal edges only (other edges will be predicted)"""
        num_messages = len(messages)

        # Get embeddings for this conversation
        message_embeddings = []
        for i, msg in enumerate(messages):
            # Find the global index of this message
            global_idx = sum(
                len(self.conversations[j]["messages"]) 
                for j in range(conv_idx)
            ) + i
            message_embeddings.append(self.embedding_dict[global_idx])

        x = torch.tensor(np.stack(message_embeddings), dtype=torch.float)

        # Only add temporal edges (bidirectional)
        edge_index = []
        for i in range(num_messages - 1):
            edge_index.extend([[i, i + 1], [i + 1, i]])

        edge_index = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if edge_index
            else torch.empty((2, 0), dtype=torch.long)
        )

        # Node features: role (0=user, 1=assistant) and position
        node_roles = torch.tensor(
            [0 if msg["role"] == "user" else 1 for msg in messages],
            dtype=torch.float,
        ).unsqueeze(1)
        node_positions = (
            torch.tensor(range(num_messages), dtype=torch.float).unsqueeze(1)
            / num_messages
        )  # Normalize

        # Additional node features
        node_attr = torch.cat([node_roles, node_positions], dim=1)

        return Data(
            x=x, edge_index=edge_index, node_attr=node_attr, num_nodes=num_messages
        )

    def create_training_targets(self, messages: List[Dict], conv_idx: int, conv_data: Dict) -> Dict:
        """Create training targets using LLM-provided dependency labels"""
        num_messages = len(messages)
        targets = {}

        # 1. Message relevance using dependency labels
        relevance_queries = []
        
        # Use LLM-provided dependencies to create supervised training pairs
        for i in range(num_messages):
            msg = messages[i]
            
            # If this message has labeled dependencies, use them
            if msg.get("is_context_dependent", False) and msg.get("depends_on_indices"):
                # Create a supervised relevance query
                relevance_queries.append({
                    "query_idx": i,
                    "context_indices": list(range(i)),  # All previous messages
                    "relevant_indices": msg["depends_on_indices"],  # Ground truth
                    "dependency_type": msg.get("dependency_type"),
                })
            elif i > 0:
                # For non-dependent messages, still include them but with no specific dependencies
                relevance_queries.append({
                    "query_idx": i,
                    "context_indices": list(range(i)),
                    "relevant_indices": [],  # No specific dependencies
                    "dependency_type": None,
                })

        targets["relevance_queries"] = relevance_queries

        # 2. Context-dependent message insertion using labeled data
        # Find all context-dependent user messages
        context_dependent_messages = []
        for i, msg in enumerate(messages):
            if (msg["role"] == "user" and 
                msg.get("is_context_dependent", False) and 
                i > 0):  # Not the first message
                
                # Get the message embedding
                global_idx = sum(
                    len(self.conversations[j]["messages"]) 
                    for j in range(conv_idx)
                ) + i
                
                context_dependent_messages.append({
                    "idx": i,
                    "text": msg["text"],
                    "embedding": self.embedding_dict[global_idx],
                    "dependency_type": msg.get("dependency_type"),
                    "depends_on": msg.get("depends_on_indices", [])
                })
        
        # Use one of the context-dependent messages for insertion training
        if context_dependent_messages and random.random() < 0.5:
            candidate = random.choice(context_dependent_messages)
            
            # Create a more informed position distribution
            position_scores = torch.zeros(num_messages)
            
            # The actual position gets highest score
            position_scores[candidate["idx"]] = 10.0
            
            # Positions after messages this one depends on get moderate scores
            for dep_idx in candidate["depends_on"]:
                if dep_idx < num_messages - 1:
                    position_scores[dep_idx + 1] = 5.0
            
            # Positions after assistant messages get small boost
            for i in range(1, num_messages):
                if messages[i-1]["role"] == "assistant":
                    position_scores[i] += 1.0
            
            # Convert to probability distribution
            position_probs = F.softmax(position_scores, dim=0)
            
            targets["vague_query"] = {
                "query_embedding": candidate["embedding"],
                "query_text": candidate["text"],
                "position_distribution": position_probs.numpy(),
                "all_positions": list(range(num_messages)),
                "dependency_type": candidate["dependency_type"],
                "is_labeled": True  # Flag that this uses LLM labels
            }

        return targets


class ConversationGNN(nn.Module):
    """Graph Neural Network for conversation understanding"""

    def __init__(
        self, input_dim=384, hidden_dim=256, output_dim=128, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim

        # Incorporate node attributes (role, position)
        self.node_encoder = nn.Linear(input_dim + 2, input_dim)

        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(
            hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout
        )

        # Task-specific heads
        # Attention scoring: given a query message, score its relevance to context messages
        self.attention_scorer = nn.Sequential(
            nn.Linear(output_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.query_insertion_head = nn.Sequential(
            nn.Linear(output_dim + input_dim, 128),  # Node embedding + query embedding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.dropout = dropout

    def forward(self, x, edge_index, node_attr=None, batch=None):
        # Incorporate node attributes
        if node_attr is not None:
            x = torch.cat([x, node_attr], dim=-1)
            x = self.node_encoder(x)

        # GAT layers
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return x

    def score_relevance(self, query_embedding, context_embedding, temperature=1.0):
        """Score the relevance of a context message to a query message

        Args:
            query_embedding: Embedding of the query message
            context_embedding: Embedding of the context message
            temperature: Temperature for controlling attention spread (default=1.0)
                        Lower values (0.5) = more focused on recent messages
                        Higher values (2.0+) = more uniform attention across history
        """
        combined = torch.cat([query_embedding, context_embedding], dim=-1)
        score = self.attention_scorer(combined)
        return score / temperature

    def score_query_insertion(self, node_embedding, query_embedding):
        """Score how well a query fits after a node"""
        combined = torch.cat([node_embedding, query_embedding], dim=-1)
        return torch.sigmoid(self.query_insertion_head(combined))


class ConversationGNNTrainer:
    """Trainer for the conversation GNN"""

    def __init__(
        self,
        model: ConversationGNN,
        device="cuda",
        learning_rate=1e-4,
        accumulation_steps=1,
        is_distributed=False,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.accumulation_steps = accumulation_steps
        self.is_distributed = is_distributed

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Only show progress bar on rank 0
        is_main_process = not self.is_distributed or (
            dist.is_initialized() and dist.get_rank() == 0
        )
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            file=sys.stdout,
            dynamic_ncols=True,
            disable=not is_main_process,
        )

        for batch_idx, (graph_batch, targets_batch) in enumerate(progress_bar):
            # Move to device
            graph_batch = graph_batch.to(self.device)

            # Forward pass
            node_embeddings = self.model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.node_attr,
                graph_batch.batch,
            )

            # Calculate losses for different tasks
            losses = []
            metrics = {}

            # 1. Message relevance loss
            if "relevance_queries" in targets_batch[0]:
                relevance_loss = self.compute_relevance_loss(
                    node_embeddings, graph_batch, targets_batch
                )
                if relevance_loss is not None:
                    losses.append(relevance_loss)
                    metrics["relevance_loss"] = relevance_loss.item()

            # 3. Vague query insertion loss
            if any("vague_query" in t for t in targets_batch):
                query_loss, query_metrics = self.compute_query_insertion_loss(
                    node_embeddings, graph_batch, targets_batch
                )
                if query_loss is not None:
                    losses.append(query_loss)
                    # Add all query metrics with prefix
                    for k, v in query_metrics.items():
                        metrics[f"query_{k}"] = v

            # Combine losses
            if losses:
                total_batch_loss = sum(losses) / len(losses)
                total_batch_loss = total_batch_loss / self.accumulation_steps

                # Backward pass
                total_batch_loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += total_batch_loss.item() * self.accumulation_steps
                num_batches += 1

                # Update progress bar
                if is_main_process:
                    postfix = {
                        "loss": total_batch_loss.item() * self.accumulation_steps
                    }
                    postfix.update(metrics)
                    progress_bar.set_postfix(postfix)
                    sys.stdout.flush()  # Force output

        # Ensure final gradients are applied
        if (batch_idx + 1) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()

        return total_loss / max(num_batches, 1)

    def compute_relevance_loss(self, node_embeddings, graph_batch, targets_batch):
        """Compute loss for learning message relevance patterns using LLM labels"""
        total_loss = 0.0
        num_queries = 0

        # Process each graph in the batch
        for i, targets in enumerate(targets_batch):
            if "relevance_queries" not in targets:
                continue

            # Get node embeddings for this graph
            mask = graph_batch.batch == i
            graph_embeddings = node_embeddings[mask]

            for query_info in targets["relevance_queries"]:
                query_idx = query_info["query_idx"]
                context_indices = query_info["context_indices"]
                relevant_indices = query_info.get("relevant_indices", [])

                if query_idx >= len(graph_embeddings) or not context_indices:
                    continue

                query_emb = graph_embeddings[query_idx]

                # Compute relevance scores for all context messages
                scores = []
                for ctx_idx in context_indices:
                    if ctx_idx < len(graph_embeddings):
                        ctx_emb = graph_embeddings[ctx_idx]
                        # Handle DDP wrapper
                        model = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )
                        score = model.score_relevance(
                            query_emb.unsqueeze(0), ctx_emb.unsqueeze(0)
                        )
                        scores.append(score)

                if scores:
                    # Stack scores and apply softmax to get attention distribution
                    scores = torch.cat(scores, dim=0)
                    attention_weights = F.softmax(scores, dim=0)

                    # Create target distribution based on LLM labels
                    if relevant_indices:
                        # Supervised: use LLM-provided dependencies
                        target_weights = torch.zeros(len(scores), device=self.device)
                        for rel_idx in relevant_indices:
                            if rel_idx in context_indices:
                                pos = context_indices.index(rel_idx)
                                target_weights[pos] = 1.0
                        
                        # Normalize to create a distribution
                        if target_weights.sum() > 0:
                            target_weights = target_weights / target_weights.sum()
                        else:
                            # Fallback to uniform if no valid indices
                            target_weights = torch.ones_like(target_weights) / len(target_weights)
                    else:
                        # For non-dependent messages, use a weak recency bias
                        # This encourages looking at recent messages by default
                        position_weights = torch.arange(
                            len(scores), 0, -1, dtype=torch.float, device=self.device
                        )
                        # Softer bias for non-dependent messages
                        position_weights = position_weights ** 0.5  
                        target_weights = position_weights / position_weights.sum()

                    # KL divergence between learned attention and target distribution
                    loss = F.kl_div(
                        attention_weights.log(), target_weights, reduction="batchmean"
                    )
                    total_loss += loss
                    num_queries += 1

        if num_queries > 0:
            return total_loss / num_queries

        return None

    def compute_query_insertion_loss(self, node_embeddings, graph_batch, targets_batch):
        """Compute loss for vague query insertion using soft labels"""
        total_loss = 0.0
        total_entropy = 0.0
        total_top1_prob = 0.0
        total_prior_alignment = 0.0
        num_samples = 0

        for i, targets in enumerate(targets_batch):
            if "vague_query" not in targets:
                continue

            mask = graph_batch.batch == i
            graph_embeddings = node_embeddings[mask]

            # Get query embedding
            query_emb = torch.tensor(
                targets["vague_query"]["query_embedding"],
                device=self.device,
                dtype=torch.float,
            )

            # Get position distribution (soft labels)
            position_dist = torch.tensor(
                targets["vague_query"]["position_distribution"],
                device=self.device,
                dtype=torch.float,
            )

            # Score all positions
            position_scores = []
            for pos in targets["vague_query"]["all_positions"]:
                if pos < len(graph_embeddings):
                    # Handle DDP wrapper
                    model = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )
                    score = model.score_query_insertion(
                        graph_embeddings[pos].unsqueeze(0), query_emb.unsqueeze(0)
                    )
                    position_scores.append(score.squeeze())

            if position_scores:
                # Stack scores and apply softmax to get predicted distribution
                pred_scores = torch.stack(position_scores)
                pred_dist = F.softmax(pred_scores, dim=0)
                
                # KL divergence between predicted and target distributions
                # This allows the model to learn from soft supervision
                loss = F.kl_div(
                    pred_dist.log(), 
                    position_dist[:len(position_scores)], 
                    reduction='batchmean'
                )
                
                # Compute meaningful metrics
                # 1. Entropy of predicted distribution (lower = more confident)
                entropy = -(pred_dist * pred_dist.log()).sum()
                
                # 2. Top-1 probability (higher = more confident in best position)
                top1_prob = pred_dist.max()
                
                # 3. Alignment with prior (cosine similarity between distributions)
                # This shows if model is learning patterns beyond the simple prior
                prior_alignment = F.cosine_similarity(
                    pred_dist.unsqueeze(0), 
                    position_dist[:len(position_scores)].unsqueeze(0)
                ).item()
                
                total_loss += loss
                total_entropy += entropy.item()
                total_top1_prob += top1_prob.item()
                total_prior_alignment += prior_alignment
                num_samples += 1

        if num_samples > 0:
            avg_loss = total_loss / num_samples
            avg_entropy = total_entropy / num_samples
            avg_top1_prob = total_top1_prob / num_samples
            avg_alignment = total_prior_alignment / num_samples
            
            # Return loss and a dict of metrics
            metrics = {
                "entropy": avg_entropy,
                "top1_prob": avg_top1_prob,
                "prior_align": avg_alignment
            }
            return avg_loss, metrics

        return None, {}

    def validate(self, dataloader: DataLoader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Only show progress bar on rank 0
        is_main_process = not self.is_distributed or (
            dist.is_initialized() and dist.get_rank() == 0
        )
        with torch.no_grad():
            for graph_batch, targets_batch in tqdm(
                dataloader,
                desc="Validation",
                file=sys.stdout,
                dynamic_ncols=True,
                disable=not is_main_process,
            ):
                graph_batch = graph_batch.to(self.device)

                node_embeddings = self.model(
                    graph_batch.x,
                    graph_batch.edge_index,
                    graph_batch.node_attr,
                    graph_batch.batch,
                )

                losses = []

                # Compute validation losses
                if "relevance_queries" in targets_batch[0]:
                    relevance_loss = self.compute_relevance_loss(
                        node_embeddings, graph_batch, targets_batch
                    )
                    if relevance_loss is not None:
                        losses.append(relevance_loss)

                if any("vague_query" in t for t in targets_batch):
                    query_loss, query_metrics = self.compute_query_insertion_loss(
                        node_embeddings, graph_batch, targets_batch
                    )
                    if query_loss is not None:
                        losses.append(query_loss)

                if losses:
                    total_batch_loss = sum(losses) / len(losses)
                    total_loss += total_batch_loss.item()
                    num_batches += 1

        return total_loss / max(num_batches, 1)


def load_conversations(file_paths: List[str]) -> List[Dict]:
    """Load conversations from JSON files"""
    all_conversations = []

    for file_path in file_paths:
        logger.info(f"Loading conversations from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Expect list of conversation objects with messages and metadata
        all_conversations.extend(data)

    return all_conversations


def collate_fn(batch):
    """Custom collate function for geometric data"""
    graphs, targets = zip(*batch)
    batched_graph = Batch.from_data_list(graphs)
    return batched_graph, list(targets)


def setup_distributed():
    """Setup distributed training - compatible with torchrun"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train Conversation GNN")
    # Default to looking for conversations in the raw data directory
    default_data_path = str(RAW_DATA_DIR / "conversations.json")
    parser.add_argument(
        "--data-paths",
        nargs="+",
        default=[default_data_path],
        help="Paths to conversation JSON files",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model to use",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension for GNN"
    )
    parser.add_argument(
        "--output-dim", type=int, default=DEFAULT_OUTPUT_DIM, help="Output dimension for GNN"
    )
    parser.add_argument(
        "--num-heads", type=int, default=DEFAULT_NUM_HEADS, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs"
    )
    # num_workers removed - always 0 since we use pre-computed embeddings
    parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Weights & Biases project name"
    )
    # Distributed training is auto-detected from environment variables

    args = parser.parse_args()

    # Setup distributed training if available
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # Set device
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )

    # Only log on main process
    is_main_process = not is_distributed or rank == 0

    # Initialize wandb (optional)
    if args.wandb_project and is_main_process:
        wandb.init(project=args.wandb_project, config=vars(args))

    if is_main_process:
        logger.info(f"Using device: {device}")
        if is_distributed:
            logger.info(f"Distributed training: rank {rank}/{world_size}")

    # Load data
    conversations = load_conversations(args.data_paths)
    if is_main_process:
        logger.info(f"Loaded {len(conversations)} conversations")

    # Split data
    split_idx = int(0.9 * len(conversations))
    train_conversations = conversations[:split_idx]
    val_conversations = conversations[split_idx:]

    # Create encoder once and compute all embeddings
    if is_main_process:
        logger.info("Creating encoder and computing embeddings...")

    encoder = SentenceTransformer(args.encoder_model)
    embedding_dim = encoder.get_sentence_embedding_dimension()

    # Collect all messages from both train and val
    all_messages = []
    for conv_data in conversations:
        messages = conv_data.get("messages", conv_data)  # Handle both formats
        for msg in messages:
            all_messages.append(msg["text"])

    # Compute embeddings once for all messages
    show_progress = is_main_process
    all_embeddings = encoder.encode(
        all_messages,
        batch_size=256,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    # Split embeddings for train and val
    train_msg_count = sum(
        len(conv["messages"]) for conv in train_conversations
    )
    train_embeddings = all_embeddings[:train_msg_count]
    val_embeddings = all_embeddings[train_msg_count:]

    # Create datasets with pre-computed embeddings
    if is_main_process:
        logger.info("Creating datasets...")
    train_dataset = ConversationDataset(
        train_conversations, train_embeddings, embedding_dim
    )
    val_dataset = ConversationDataset(
        val_conversations, val_embeddings, embedding_dim
    )

    if is_main_process:
        logger.info(f"Using embedding dimension: {embedding_dim}")

    # Delete encoder to free memory
    del encoder

    # Create distributed samplers if needed
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Always 0 - we use pre-computed embeddings
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Always 0 - we use pre-computed embeddings
        pin_memory=True,
    )

    # Initialize model
    if is_main_process:
        logger.info("Initializing model...")
    model = ConversationGNN(
        input_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    # Wrap model for distributed training
    if is_distributed:
        model = DDP(model.to(device), device_ids=[local_rank])

    # Initialize trainer
    trainer = ConversationGNNTrainer(
        model,
        device=device,
        learning_rate=args.learning_rate,
        accumulation_steps=args.accumulation_steps,
        is_distributed=is_distributed,
    )

    # Training loop
    if is_main_process:
        logger.info("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_loss = trainer.validate(val_loader)

        if is_main_process:
            logger.info(
                f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

        # Save best model (only on rank 0 for distributed)
        if val_loss < best_val_loss and is_main_process:
            best_val_loss = val_loss
            model_to_save = model.module if hasattr(model, "module") else model
            
            # Save model checkpoint
            model_path = Path(args.model_save_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "embedding_dim": embedding_dim,
                },
                model_path,
            )
            logger.info(f"Saved best model with val loss {val_loss:.4f}")
            
        # Save training checkpoint periodically
        if is_main_process and epoch % 10 == 0:
            checkpoint_path = TRAINING_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            logger.info(f"Saved training checkpoint at epoch {epoch}")

        if args.wandb_project and is_main_process:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

    if is_main_process:
        logger.info("Training complete!")

    if is_distributed:
        cleanup_distributed()




if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    # Train
    main()
