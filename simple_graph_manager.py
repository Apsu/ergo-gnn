#!/usr/bin/env python3
"""
Simple Unified Graph Manager for GNN-based Conversation Retrieval
Maintains a PyTorch Geometric graph with temporal and semantic edges
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleGraphManager:
    """
    Simple conversation graph manager using PyTorch Geometric
    
    - Maintains unified PyTorch Geometric graph
    - Temporal edges between consecutive messages
    - Semantic edges based on similarity threshold
    - Direct node addition (no temporary queries)
    - GNN handles all scoring logic
    """
    
    def __init__(self, 
                 gnn_model,
                 encoder_model='sentence-transformers/all-MiniLM-L6-v2',
                 semantic_threshold=0.75,
                 device='cuda'):
        """
        Args:
            gnn_model: Trained ConversationGNN model
            encoder_model: Sentence transformer model name
            semantic_threshold: Cosine similarity threshold for semantic edges
            device: Device for computation (torch.device or string)
        """
        # Handle device parameter
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        self.gnn_model = gnn_model.to(self.device)
        self.gnn_model.eval()
        # Initialize encoder with progress bars disabled
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer warnings
        self.encoder = SentenceTransformer(encoder_model)
        self.semantic_threshold = semantic_threshold
        
        # Graph components
        self.node_features = []  # List of embedding tensors
        self.node_attrs = []     # List of [role, normalized_position]
        self.edge_list = []      # List of [source, target]
        self.edge_types = []     # List of edge types ('temporal' or 'semantic')
        self.edge_weights = []   # List of edge weights
        
        # Message storage
        self.messages = []
        
        # Cache for GNN embeddings (invalidated on graph change)
        self._cached_embeddings = None
        self._cache_valid = False
        
        logger.info(f"Initialized SimpleGraphManager with semantic threshold={semantic_threshold}")
    
    def add_message(self, role: str, text: str) -> int:
        """
        Add a message to the graph
        
        Returns:
            Message ID
        """
        msg_id = len(self.messages)
        
        # Store message
        self.messages.append({
            'id': msg_id,
            'role': role,
            'text': text,
            'timestamp': datetime.now().isoformat()
        })
        
        # Encode message (disable progress bar for cleaner output)
        embedding = self.encoder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device)
        self.node_features.append(embedding_tensor)
        
        # Add node attributes [role, normalized_position]
        role_int = 0 if role == 'user' else 1
        normalized_pos = msg_id / max(1, msg_id)
        self.node_attrs.append([role_int, normalized_pos])
        
        # Add temporal edges with previous message (bidirectional)
        if msg_id > 0:
            self.edge_list.append([msg_id - 1, msg_id])
            self.edge_list.append([msg_id, msg_id - 1])
            self.edge_types.extend(['temporal', 'temporal'])
            self.edge_weights.extend([1.0, 1.0])
        
        # Add semantic edges
        if msg_id > 0:
            self._add_semantic_edges(msg_id, embedding_tensor)
        
        # Invalidate cache
        self._cache_valid = False
        
        logger.info(f"Added message {msg_id} ({role}). Graph has {len(self.edge_list)} edges.")
        return msg_id
    
    def _add_semantic_edges(self, new_idx: int, new_embedding: torch.Tensor):
        """Add semantic edges based on similarity"""
        new_norm = F.normalize(new_embedding.unsqueeze(0), p=2, dim=1)
        
        semantic_edges_added = 0
        max_similarity = 0.0
        
        for i in range(new_idx):
            other_norm = F.normalize(self.node_features[i].unsqueeze(0), p=2, dim=1)
            similarity = torch.cosine_similarity(new_norm, other_norm).item()
            max_similarity = max(max_similarity, similarity)
            
            if similarity >= self.semantic_threshold:
                # Add bidirectional semantic edge
                self.edge_list.append([new_idx, i])
                self.edge_list.append([i, new_idx])
                self.edge_types.extend(['semantic', 'semantic'])
                self.edge_weights.extend([similarity, similarity])
                semantic_edges_added += 1
                logger.debug(f"Added semantic edge between {new_idx} and {i} (sim={similarity:.3f})")
        
        # Log summary at INFO level to diagnose
        if semantic_edges_added > 0:
            logger.info(f"Added {semantic_edges_added} semantic edges for message {new_idx} (threshold={self.semantic_threshold})")
        elif new_idx > 0:
            logger.info(f"No semantic edges for message {new_idx} (max_sim={max_similarity:.3f}, threshold={self.semantic_threshold})")
    
    def build_graph_data(self) -> Data:
        """Build PyTorch Geometric Data object from current graph"""
        if len(self.node_features) == 0:
            # Empty graph
            return Data(
                x=torch.empty((0, 384), device=self.device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                edge_weight=torch.empty((0,), dtype=torch.float32, device=self.device),
                node_attr=torch.empty((0, 2), dtype=torch.float32, device=self.device)
            )
        
        # Stack features
        x = torch.stack(self.node_features)
        
        # Convert edges
        if self.edge_list:
            edge_index = torch.tensor(self.edge_list, dtype=torch.long, device=self.device).t().contiguous()
            edge_weight = torch.tensor(self.edge_weights, dtype=torch.float32, device=self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_weight = torch.empty((0,), dtype=torch.float32, device=self.device)
        
        # Convert attributes
        node_attr = torch.tensor(self.node_attrs, dtype=torch.float32, device=self.device)
        
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, node_attr=node_attr)
    
    def get_gnn_embeddings(self, force_recompute: bool = False) -> torch.Tensor:
        """Get GNN embeddings for all messages (cached)"""
        if self._cache_valid and not force_recompute and self._cached_embeddings is not None:
            return self._cached_embeddings
        
        graph_data = self.build_graph_data()
        
        with torch.no_grad():
            embeddings = self.gnn_model(graph_data.x, graph_data.edge_index, graph_data.node_attr)
        
        self._cached_embeddings = embeddings
        self._cache_valid = True
        
        return embeddings
    
    def score_all_pairs(self, temperature: float = 1.5) -> torch.Tensor:
        """
        Compute pairwise relevance scores between all messages
        
        Returns:
            Score matrix of shape [n_messages, n_messages]
        """
        embeddings = self.get_gnn_embeddings()
        n = len(self.messages)
        
        if n == 0:
            return torch.empty((0, 0), device=self.device)
        
        scores = torch.zeros((n, n), device=self.device)
        
        with torch.no_grad():
            for i in range(n):
                for j in range(n):
                    if i != j:
                        score = self.gnn_model.score_relevance(
                            embeddings[i].unsqueeze(0),
                            embeddings[j].unsqueeze(0),
                            temperature=temperature
                        )
                        scores[i, j] = score
        
        return scores
    
    def get_relevant_context(self, query_idx: int, temperature: float = 1.5,
                           top_k: int = 10, min_score: float = 0.001,
                           min_recent: int = 3) -> Tuple[List[Dict], List[int]]:
        """
        Get relevant context for a message that's already in the graph
        
        Args:
            query_idx: Index of the query message
            temperature: Temperature for attention
            top_k: Maximum number of context messages
            min_score: Minimum attention score threshold
            min_recent: Minimum number of recent messages to include
            
        Returns:
            (messages, indices) in chronological order, excluding the query message itself
        """
        if query_idx >= len(self.messages):
            return [], []
        
        # Always include recent messages for coherence
        recent_indices = set()
        if query_idx > 0:
            # Get the most recent messages before query
            start_idx = max(0, query_idx - min_recent)
            recent_indices = set(range(start_idx, query_idx))
        
        # Get embeddings and score for additional relevant messages
        embeddings = self.get_gnn_embeddings()
        
        raw_scores = []
        indices = []
        with torch.no_grad():
            query_emb = embeddings[query_idx]
            for i in range(len(self.messages)):
                if i != query_idx and i not in recent_indices:  # Don't score self or already included recent
                    score = self.gnn_model.score_relevance(
                        query_emb.unsqueeze(0),
                        embeddings[i].unsqueeze(0),
                        temperature=temperature
                    )
                    raw_scores.append(score)
                    indices.append(i)
        
        # Apply softmax to get probabilities
        if raw_scores:
            scores_tensor = torch.cat(raw_scores)
            probs = F.softmax(scores_tensor, dim=0)
            scores = [(prob.item(), idx) for prob, idx in zip(probs, indices)]
        else:
            scores = []
        
        # Sort by score and get high-scoring messages
        scores.sort(reverse=True)
        gnn_selected = []
        remaining_slots = top_k - len(recent_indices)
        
        # Debug: log top scores
        if scores:
            top_scores = scores[:5]  # Top 5 for debugging
            logger.debug(f"Top GNN scores: {[(f'{score:.3f}', idx) for score, idx in top_scores]}")
        
        for score, idx in scores:
            if score > min_score and len(gnn_selected) < remaining_slots:
                gnn_selected.append(idx)
        
        # Combine recent + GNN-selected messages
        all_indices = sorted(list(recent_indices) + gnn_selected)
        
        # Track how many we're adding in fill phase
        fill_count = 0
        
        # If we still have room, add more recent messages
        if len(all_indices) < top_k and query_idx > min_recent:
            extra_recent = max(0, query_idx - top_k)
            for i in range(extra_recent, query_idx):
                if i not in all_indices:
                    all_indices.append(i)
                    fill_count += 1
            all_indices.sort()
            all_indices = all_indices[-top_k:]  # Keep only the last top_k
        
        # Get messages
        relevant_messages = [self.messages[idx] for idx in all_indices]
        
        # Better logging
        actual_gnn_in_final = len([idx for idx in all_indices if idx not in recent_indices and idx < query_idx - min_recent])
        logger.info(f"Context selection: {len(recent_indices)} recent + {actual_gnn_in_final} GNN-selected + {fill_count} fill = {len(all_indices)} total")
        
        return relevant_messages, all_indices
    
    def build_llm_messages(self, query_idx: int, system_prompt: str, 
                          temperature: float = 1.5, context_window: int = 10,
                          min_recent: int = 3) -> List[Dict[str, str]]:
        """
        Build message array for LLM with context + query at the end
        
        Args:
            query_idx: Index of the user's query message
            system_prompt: System prompt for the LLM
            temperature: Temperature for GNN attention
            context_window: Maximum number of context messages
            min_recent: Minimum recent messages to include for coherence
            
        Returns:
            List of message dicts ready for LLM API
        """
        # Get relevant context (excluding the query itself)
        context_messages, context_indices = self.get_relevant_context(
            query_idx, 
            temperature=temperature,
            top_k=context_window,
            min_recent=min_recent
        )
        
        # Build message list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context messages
        for msg in context_messages:
            messages.append({
                "role": msg['role'],
                "content": msg['text']
            })
        
        # Add the query message at the end
        query_msg = self.messages[query_idx]
        messages.append({
            "role": query_msg['role'],
            "content": query_msg['text']
        })
        
        logger.info(f"Built LLM messages: {len(context_messages)} context + 1 query")
        return messages
    
    def get_graph_stats(self) -> Dict:
        """Get current graph statistics"""
        temporal_edges = sum(1 for t in self.edge_types if t == 'temporal')
        semantic_edges = sum(1 for t in self.edge_types if t == 'semantic')
        
        return {
            'num_messages': len(self.messages),
            'num_edges': len(self.edge_list),
            'temporal_edges': temporal_edges,
            'semantic_edges': semantic_edges,
            'avg_semantic_edges_per_node': semantic_edges / max(1, len(self.messages))
        }


def demo_usage():
    """Demonstrate simple usage pattern"""
    print("\nSIMPLE GRAPH MANAGER USAGE:")
    print("-" * 40)
    print("1. Add user message to graph")
    print("2. Find relevant context using GNN")
    print("3. Send context + user message to LLM")
    print("4. Add LLM response to graph")
    print("5. Repeat")
    print("\nNo temporary nodes, no heuristics!")
    print("Just: add → score → retrieve → add")


if __name__ == "__main__":
    demo_usage()