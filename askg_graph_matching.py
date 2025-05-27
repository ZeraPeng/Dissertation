import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import ipdb
from s_model import (reparameterize)


class ASKG:
    """
    Action Skeleton Knowledge Graph (ASKG) class for representing heterogeneous relationships
    between class labels, objects, and sub-actions in action recognition tasks.
    
    The knowledge graph contains three types of nodes:
    - class: Action class labels
    - object: Objects related to actions
    - sub-action: Sub-actions that compose complex actions
    
    And three types of edges:
    - object: Undirected edges between classes and related objects
    - sub-action: Undirected edges between classes and related sub-actions
    - precedes: Directed edges representing temporal order between sub-actions
    """
    
    def __init__(self, askg_mapping: Dict, embedding_dim: int = 512):
        """
        Initialize the ASKG with predefined mappings.
        
        Args:
            askg_mapping: Dictionary containing cls2obj and cls2sa mappings
            embedding_dim: Dimension of semantic embeddings for nodes
        """
        self.embedding_dim = embedding_dim
        self.askg_mapping = askg_mapping
        
        # Initialize node dictionaries
        self.class_nodes = {}
        self.object_nodes = {}
        self.subaction_nodes = {}
        
        # Edge lists for different types
        self.object_edges = []  # (class_idx, object_idx)
        self.subaction_edges = []  # (class_idx, subaction_idx)
        self.precedes_edges = []  # (subaction_idx_1, subaction_idx_2)
        
        # Node embeddings
        self.class_embeddings = None
        self.object_embeddings = None
        self.subaction_embeddings = None
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the knowledge graph from askg_mapping."""
        cls2obj = self.askg_mapping.get('cls2obj', {})
        cls2sa = self.askg_mapping.get('cls2sa', {})
        
        # Build class nodes
        all_classes = set(cls2obj.keys()) | set(cls2sa.keys())
        for i, cls in enumerate(sorted(all_classes)):
            self.class_nodes[cls] = i
        
        # Build object nodes and edges
        all_objects = set()
        for cls, objects in cls2obj.items():
            all_objects.update(objects)
        
        for i, obj in enumerate(sorted(all_objects)):
            self.object_nodes[obj] = i
        
        # Create object edges
        for cls, objects in cls2obj.items():
            cls_idx = self.class_nodes[cls]
            for obj in objects:
                obj_idx = self.object_nodes[obj]
                self.object_edges.append((cls_idx, obj_idx))
        
        # Build sub-action nodes and edges
        all_subactions = set()
        for cls, subactions in cls2sa.items():
            all_subactions.update(subactions)
        
        for i, sa in enumerate(sorted(all_subactions)):
            self.subaction_nodes[sa] = i
        
        # Create sub-action edges
        for cls, subactions in cls2sa.items():
            cls_idx = self.class_nodes[cls]
            for sa in subactions:
                sa_idx = self.subaction_nodes[sa]
                self.subaction_edges.append((cls_idx, sa_idx))
            
            # Create temporal precedes edges between consecutive sub-actions
            for i in range(len(subactions) - 1):
                sa1_idx = self.subaction_nodes[subactions[i]]
                sa2_idx = self.subaction_nodes[subactions[i + 1]]
                self.precedes_edges.append((sa1_idx, sa2_idx))
    
    def load_embeddings(self, class_emb: torch.Tensor, object_emb: torch.Tensor, 
                       subaction_emb: torch.Tensor):
        """
        Load pre-trained semantic embeddings for nodes.
        
        Args:
            class_emb: Tensor of shape [num_classes, embedding_dim]
            object_emb: Tensor of shape [num_objects, embedding_dim]
            subaction_emb: Tensor of shape [num_subactions, embedding_dim]
        """
        self.class_embeddings = class_emb
        self.object_embeddings = object_emb
        self.subaction_embeddings = subaction_emb
    
    def get_adjacency_matrices(self) -> Dict[str, torch.Tensor]:
        """
        Get adjacency matrices for different edge types.
        
        Returns:
            Dictionary containing adjacency matrices for each edge type
        """
        num_classes = len(self.class_nodes)
        num_objects = len(self.object_nodes)
        num_subactions = len(self.subaction_nodes)
        
        # Object adjacency matrix (classes x objects)
        obj_adj = torch.zeros(num_classes, num_objects)
        for cls_idx, obj_idx in self.object_edges:
            obj_adj[cls_idx, obj_idx] = 1
        
        # Sub-action adjacency matrix (classes x sub-actions)
        sa_adj = torch.zeros(num_classes, num_subactions)
        for cls_idx, sa_idx in self.subaction_edges:
            sa_adj[cls_idx, sa_idx] = 1
        
        # Precedes adjacency matrix (sub-actions x sub-actions)
        precedes_adj = torch.zeros(num_subactions, num_subactions)
        for sa1_idx, sa2_idx in self.precedes_edges:
            precedes_adj[sa1_idx, sa2_idx] = 1
        
        return {
            'object': obj_adj,
            'subaction': sa_adj,
            'precedes': precedes_adj
        }
    
    def get_node_info(self) -> Dict:
        """Get information about all nodes in the graph."""
        return {
            'classes': self.class_nodes,
            'objects': self.object_nodes,
            'subactions': self.subaction_nodes,
            'num_classes': len(self.class_nodes),
            'num_objects': len(self.object_nodes),
            'num_subactions': len(self.subaction_nodes)
        }


class MultiHeadGraphAttention(nn.Module):
    """Multi-head attention mechanism for graph nodes."""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0
        
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, input_dim]
            key: Key tensor [batch_size, seq_len, input_dim]
            value: Value tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Attention output tensor
        """
        batch_size = query.size(0)  # 1
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Ensure key and value have the same sequence length
        assert seq_len_k == seq_len_v, f"Key and value must have same sequence length, got {seq_len_k} and {seq_len_v}"
        
        # Linear projections
        q = self.query(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.key(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.value(value).view(batch_size, seq_len_v, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # torch.Size([1, 4, 4, 12])
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)    # torch.Size([1, 4, 4, 12])
        attn_output = torch.matmul(attn_weights, v)     # torch.Size([1, 4, 4, 24])
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, -1)      # torch.Size([1, 4, 96])
        
        return self.out_proj(attn_output)


class AdaptiveFusion(nn.Module):
    """Adaptive fusion module for combining different similarity scores."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, path_features: torch.Tensor, 
                graph_features: torch.Tensor) -> torch.Tensor:
        """
        Adaptively fuse path-level and graph-level features.
        
        Args:
            path_features: Path-level similarity features
            graph_features: Graph-level similarity features
            
        Returns:
            Fused features
        """
        combined = torch.cat([path_features, graph_features], dim=-1)
        weights = self.weight_net(combined)
        fused = weights[:, 0:1] * path_features + weights[:, 1:2] * graph_features
        return fused


class GraphMatcher(nn.Module):
    """
    Graph Matching Module for temporal representation matching with knowledge graph.
    
    This module implements the graph matching procedure described in the documentation:
    1. Node Alignment: Find interested nodes and calculate similarity scores
    2. Graph Alignment: Build sample graph and align with interested subgraph
    3. Integration: Calculate total similarity score for classification
    """
    
    def __init__(self, askg: ASKG, embedding_dim: int = 96, top_k: int = 5, 
                 num_classes: int = 5):
        super().__init__()
        self.askg = askg
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.num_classes = num_classes
        
        # Multi-head attention for node alignment
        self.node_attention = MultiHeadGraphAttention(
            input_dim=embedding_dim, 
            output_dim=embedding_dim, 
            num_heads=8
        )
        
        # Graph attention for graph alignment
        self.graph_attention = MultiHeadGraphAttention(
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            num_heads=4
        )
        
        # Adaptive fusion module
        self.adaptive_fusion = AdaptiveFusion(input_dim=embedding_dim)
        
        # Final classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def node_alignment(self, temporal_repr: torch.Tensor, 
                      node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform node alignment to find interested nodes.
        
        Args:
            temporal_repr: Temporal representations [num_frames, embedding_dim]
            node_embeddings: Node semantic embeddings [num_nodes, embedding_dim]
            
        Returns:
            Tuple of (similarity_matrix, mapping_matrix)
        """
        num_repr, num_nodes = temporal_repr.size(0), node_embeddings.size(0)
        
        # Calculate similarity matrix
        temporal_repr_norm = F.normalize(temporal_repr, p=2, dim=1)
        node_embeddings_norm = F.normalize(node_embeddings, p=2, dim=1)
        
        similarity_matrix = torch.mm(temporal_repr_norm, node_embeddings_norm.t())
        
        # Sort to get mapping matrix (indices of top-k similar nodes for each representation)
        _, mapping_matrix = torch.topk(similarity_matrix, k=self.top_k, dim=1)
        
        return similarity_matrix, mapping_matrix

    def build_interested_graph(self, mapping_matrix: torch.Tensor, 
                            similarity_matrix: torch.Tensor,
                            node_embeddings: torch.Tensor) -> Dict:
        """
        Build interested subgraph from top-k nodes with paths in [label, [sa1, sa2, sa3]] format.
        
        Args:
            mapping_matrix: Mapping matrix from node alignment [num_representations, top_k]
            similarity_matrix: Similarity scores between representations and nodes [num_representations, num_nodes]
            node_embeddings: Node embeddings [num_nodes, embedding_dim]
            
        Returns:
            Dictionary containing interested graph information with structured paths
        """
        # Get interested node set (unique nodes from top-k selections)
        interested_nodes = torch.unique(mapping_matrix.flatten())  # e.g., tensor([9, 14, 15, 25, 42, 49, 67, 84, 107, 112, 115, 129])
        num_interested = len(interested_nodes)
        
        # Extract subgraph embeddings for interested nodes
        subgraph_node_embeddings = node_embeddings[interested_nodes]  # [num_interested, embedding_dim]
        
        # Generate path-level similarity vectors for each representation
        path_similarities = []
        for i in range(mapping_matrix.size(0)):  # For each temporal representation
            top_k_nodes = mapping_matrix[i]  # Top-k node indices for this representation
            path_sim = similarity_matrix[i, top_k_nodes]  # Similarity scores for top-k nodes
            path_similarities.append(path_sim)
        
        path_similarities = torch.stack(path_similarities, dim=0)  # [num_representations, top_k]
        
        # Extract paths from ASKG mapping in [label, [sa1, sa2, sa3]] format
        structured_paths = self._extract_structured_paths(interested_nodes)

        # Create global to local node index mapping
        global_to_local = {node_idx.item(): i for i, node_idx in enumerate(interested_nodes)}
        
        # Create adjacency matrix based on structured paths
        subgraph_adj = self._create_path_based_adjacency(interested_nodes, structured_paths)
        
        # Create node similarity matrix within the subgraph 这一块不一定有用 to be checked
        subgraph_similarity = torch.zeros(num_interested, num_interested, device=interested_nodes.device)
        for i in range(num_interested):
            for j in range(num_interested):
                if i != j:
                    # Calculate cosine similarity between node embeddings
                    node_i_emb = subgraph_node_embeddings[i]
                    node_j_emb = subgraph_node_embeddings[j]
                    sim = F.cosine_similarity(node_i_emb.unsqueeze(0), node_j_emb.unsqueeze(0))
                    subgraph_similarity[i, j] = sim
        
        return {
            'interested_nodes': interested_nodes,                    # [num_interested] - Tensor of interested node indices
            'subgraph_embeddings': subgraph_node_embeddings,        # [num_interested, embedding_dim] - Node embeddings
            'path_similarities': path_similarities,                 # [num_representations, top_k] - Similarity scores
            'structured_paths': structured_paths,                   # List of [label, [sa1, sa2, sa3]] format paths
            'subgraph_adjacency': subgraph_adj,                     # [num_interested, num_interested] - Adjacency matrix
            'subgraph_similarity': subgraph_similarity,             # [num_interested, num_interested] - Node similarity matrix
            'global_to_local_mapping': global_to_local,             # Dict - Global to local node index mapping
            'num_interested_nodes': num_interested,                 # Int - Number of interested nodes
            'num_paths': len(structured_paths)                      # Int - Number of structured paths
        }

    def _extract_structured_paths(self, interested_nodes: torch.Tensor) -> List[List]:
        """
        Extract structured paths in [label, [sa1, sa2, sa3]] format from ASKG mapping.
        Only include paths that contain at least one interested node.
        
        Args:
            interested_nodes: Tensor of interested node indices
            
        Returns:
            List of paths in [label, [sa1, sa2, sa3]] format
        """
        structured_paths = []
        cls2sa = self.askg.askg_mapping.get('cls2sa', {})
        
        # Convert interested nodes to set for efficient lookup
        interested_nodes_set = set(interested_nodes.cpu().numpy())
        
        # Create reverse mapping from subaction names to indices
        subaction_name_to_idx = {name: idx for name, idx in self.askg.subaction_nodes.items()}
        
        for class_label, subaction_sequence in cls2sa.items():
            # Convert subaction names to indices
            subaction_indices = []
            for sa_name in subaction_sequence:
                if sa_name in subaction_name_to_idx:
                    sa_idx = subaction_name_to_idx[sa_name]
                    subaction_indices.append(sa_idx)
            
            # Check if this path contains any interested nodes
            path_has_interested_nodes = any(sa_idx in interested_nodes_set for sa_idx in subaction_indices)
            
            if path_has_interested_nodes and len(subaction_indices) > 0:
                # Store in [label, [sa1, sa2, sa3]] format with both names and indices
                structured_path = {
                    'label': class_label,
                    'subaction_names': subaction_sequence,      # Original names from mapping
                    'subaction_indices': subaction_indices,    # Corresponding node indices
                    'interested_nodes_in_path': [idx for idx in subaction_indices if idx in interested_nodes_set]
                }
                structured_paths.append(structured_path)
        
        return structured_paths

    def _create_path_based_adjacency(self, interested_nodes: torch.Tensor, 
                                    structured_paths: List[Dict]) -> torch.Tensor:
        """
        Create adjacency matrix based on structured paths.
        Nodes are connected if they appear in the same path or consecutive in a sequence.
        
        Args:
            interested_nodes: Tensor of interested node indices
            structured_paths: List of structured paths
            
        Returns:
            Adjacency matrix for the subgraph
        """
        num_interested = len(interested_nodes)
        subgraph_adj = torch.zeros(num_interested, num_interested, device=interested_nodes.device)
        
        # Create global to local mapping
        global_to_local = {node_idx.item(): i for i, node_idx in enumerate(interested_nodes)}
        
        for path_info in structured_paths:
            subaction_indices = path_info['subaction_indices']
            
            # Connect nodes that appear in the same path
            for i, sa_idx_i in enumerate(subaction_indices):
                if sa_idx_i in global_to_local:
                    local_i = global_to_local[sa_idx_i]
                    
                    # Connect to all other nodes in the same path
                    for j, sa_idx_j in enumerate(subaction_indices):
                        if sa_idx_j in global_to_local and i != j:
                            local_j = global_to_local[sa_idx_j]
                            subgraph_adj[local_i, local_j] = 1
                            
                            # If nodes are consecutive in the sequence, give stronger connection
                            if abs(i - j) == 1:
                                subgraph_adj[local_i, local_j] = 2  # Stronger connection for consecutive nodes
        
        return subgraph_adj


    def graph_alignment(self, temporal_repr: torch.Tensor, interested_graph: Dict) -> Dict:
        """
        Perform graph alignment between sample graph and interested subgraph.
        
        Args:
            temporal_repr: Temporal representations [num_frames, embedding_dim]
            interested_graph: Dictionary containing interested graph info
            
        Returns:
            Dictionary containing:
            - 'structured_paths': List of path information with similarity scores
            - 'class_similarities': Dictionary mapping class labels to aggregated similarity scores
            - 'best_paths': Dictionary mapping class labels to best matching path info
            - 'graph_similarity_vector': Final similarity vector for classification [num_classes]
        """
        device = temporal_repr.device
        
        # Step 1: Aggregate temporal representation using attention
        temporal_repr_batch = temporal_repr.unsqueeze(0)  # [1, num_frames, embedding_dim]
        
        # Self-attention to aggregate temporal information
        aggregated_temporal = self.node_attention(
            query=temporal_repr_batch,
            key=temporal_repr_batch, 
            value=temporal_repr_batch
        )  # [1, num_frames, embedding_dim]
        
        # Create unified temporal representation
        unified_temporal_repr = torch.mean(aggregated_temporal.squeeze(0), dim=0)  # [embedding_dim]
        
        # Step 2: Extract path information
        structured_paths = interested_graph['structured_paths']
        subgraph_embeddings = interested_graph['subgraph_embeddings']  # [num_interested, embedding_dim]
        global_to_local = interested_graph['global_to_local_mapping']
        
        # Step 3: Calculate similarity for each individual path
        paths_with_scores = []
        class_to_paths = {}
        
        for path_idx, path_info in enumerate(structured_paths):
            class_label = path_info['label']
            subaction_indices = path_info['subaction_indices']
            subaction_names = path_info['subaction_names']
            
            # Initialize class tracking
            if class_label not in class_to_paths:
                class_to_paths[class_label] = []
            
            # Build path embedding
            path_node_embeddings = []
            valid_subactions = []
            
            for i, sa_idx in enumerate(subaction_indices):
                if sa_idx in global_to_local:
                    local_idx = global_to_local[sa_idx]
                    path_node_embeddings.append(subgraph_embeddings[local_idx])
                    valid_subactions.append({
                        'name': subaction_names[i],
                        'index': sa_idx,
                        'position_in_path': i
                    })
            
            if len(path_node_embeddings) > 0:
                # Create path embedding
                path_embedding = torch.stack(path_node_embeddings, dim=0)  # [path_length, embedding_dim]
                
                # Apply attention within the path
                path_embedding_batch = path_embedding.unsqueeze(0)  # [1, path_length, embedding_dim]
                attended_path = self.graph_attention(
                    query=path_embedding_batch,
                    key=path_embedding_batch,
                    value=path_embedding_batch
                ).squeeze(0)  # [path_length, embedding_dim]
                
                # Aggregate path to single representation
                # Cosine similarity with mean aggregation
                aggregated_path = torch.mean(attended_path, dim=0)  # [embedding_dim]

                cos_sim_mean = F.cosine_similarity(
                    unified_temporal_repr.unsqueeze(0), 
                    aggregated_path.unsqueeze(0)
                ).item()

                # Node-level similarities for detailed analysis
                node_similarities = []
                for i, node_emb in enumerate(path_node_embeddings):
                    node_sim = F.cosine_similarity(
                        unified_temporal_repr.unsqueeze(0),
                        node_emb.unsqueeze(0)
                    ).item()
                    node_similarities.append({
                        'subaction_name': valid_subactions[i]['name'],
                        'subaction_index': valid_subactions[i]['index'],
                        'position': valid_subactions[i]['position_in_path'],
                        'similarity': node_sim
                    })
                
                # Calculate final path similarity score
                final_similarity = cos_sim_mean 
                
                # Create comprehensive path information
                path_with_score = {
                    'path_index': path_idx,
                    'class_label': class_label,
                    'subaction_names': subaction_names,
                    'subaction_indices': subaction_indices,
                    'valid_subactions': valid_subactions,
                    'path_length': len(valid_subactions),
                    'similarity_scores': {
                        'final_score': final_similarity,
                        'cosine_mean': cos_sim_mean
                    },
                    'node_similarities': node_similarities,
                    'path_embedding': aggregated_path.detach()  # Store for potential future use
                }
                
                paths_with_scores.append(path_with_score)
                class_to_paths[class_label].append(path_with_score)
            else:
                # Handle paths with no valid subactions
                path_with_score = {
                    'path_index': path_idx,
                    'class_label': class_label,
                    'subaction_names': subaction_names,
                    'subaction_indices': subaction_indices,
                    'valid_subactions': [],
                    'path_length': 0,
                    'similarity_scores': {
                        'final_score': 0.0,
                        'cosine_mean': 0.0,
                        'cosine_weighted': 0.0,
                        'sequence_similarity': 0.0
                    },
                    'node_similarities': [],
                    'path_embedding': None,
                    'weighted_path_embedding': None
                }
                paths_with_scores.append(path_with_score)
                class_to_paths[class_label].append(path_with_score)
        
        # Step 4: Calculate class-level similarities and find best paths
        class_similarities = {}
        best_paths = {}
        graph_similarity_vector = torch.zeros(self.num_classes, device=device)
        
        class_names = list(class_to_paths.keys())
        for class_idx, class_label in enumerate(class_names):
            if class_idx >= self.num_classes:
                break
            
            class_paths = class_to_paths[class_label]
            valid_paths = [p for p in class_paths if p['path_length'] > 0]
            
            if len(valid_paths) > 0:
                # Get all similarity scores for this class
                scores = [p['similarity_scores']['final_score'] for p in valid_paths]
                
                # Calculate class-level aggregated similarity
                max_score = max(scores)
                mean_score = sum(scores) / len(scores)
                
                # Find best path for this class
                best_path = max(valid_paths, key=lambda x: x['similarity_scores']['final_score'])
                
                class_similarities[class_label] = {
                    'max_similarity': max_score,
                    'mean_similarity': mean_score,
                    'num_paths': len(valid_paths),
                    'all_scores': scores
                }
                
                best_paths[class_label] = best_path
                graph_similarity_vector[class_idx] = max_score  # Use max score for classification
            else:
                class_similarities[class_label] = {
                    'max_similarity': 0.0,
                    'mean_similarity': 0.0,
                    'num_paths': 0,
                    'all_scores': []
                }
                best_paths[class_label] = None
                graph_similarity_vector[class_idx] = 0.0
        
        # Step 5: Normalize the similarity vector
        if graph_similarity_vector.sum() > 0:
            graph_similarity_vector = F.softmax(graph_similarity_vector, dim=0)
        
        # Return comprehensive graph alignment results
        return {
            'structured_paths': paths_with_scores,           # List of all paths with detailed similarity info
            'class_similarities': class_similarities,        # Aggregated similarities by class
            'best_paths': best_paths,                        # Best matching path for each class
            'graph_similarity_vector': graph_similarity_vector,  # Final classification vector [num_classes]
            'num_total_paths': len(paths_with_scores),       # Total number of paths processed
            'num_valid_paths': len([p for p in paths_with_scores if p['path_length'] > 0]),  # Valid paths count
            'temporal_representation': unified_temporal_repr.detach(),  # Aggregated temporal representation
            'alignment_metadata': {
                'embedding_dim': self.embedding_dim,
                'top_k': self.top_k,
                'num_classes': self.num_classes,
                'num_temporal_frames': temporal_repr.size(0)
            }
        }

    def get_top_k_paths(graph_alignment_result: Dict, k: int = 3) -> Dict:
        """
        Utility function to get top-k paths with highest similarity scores.
        
        Args:
            graph_alignment_result: Result dictionary from graph_alignment
            k: Number of top paths to return
            
        Returns:
            Dictionary with top-k paths information
        """
        paths_with_scores = graph_alignment_result['structured_paths']
        
        # Sort paths by final similarity score
        sorted_paths = sorted(
            paths_with_scores, 
            key=lambda x: x['similarity_scores']['final_score'], 
            reverse=True
        )
        
        top_k_paths = sorted_paths[:k]
        
        return {
            'top_k_paths': top_k_paths,
            'top_k_scores': [p['similarity_scores']['final_score'] for p in top_k_paths],
            'top_k_classes': [p['class_label'] for p in top_k_paths],
            'score_statistics': {
                'max_score': max([p['similarity_scores']['final_score'] for p in paths_with_scores]),
                'min_score': min([p['similarity_scores']['final_score'] for p in paths_with_scores]),
                'mean_score': sum([p['similarity_scores']['final_score'] for p in paths_with_scores]) / len(paths_with_scores)
            }
        }

    def analyze_path_similarities(graph_alignment_result: Dict) -> Dict:
        """
        Utility function to analyze path similarities in detail.
        
        Args:
            graph_alignment_result: Result dictionary from graph_alignment
            
        Returns:
            Detailed analysis of path similarities
        """
        paths_with_scores = graph_alignment_result['structured_paths']
        class_similarities = graph_alignment_result['class_similarities']
        
        analysis = {
            'per_class_analysis': {},
            'overall_statistics': {},
            'similarity_distribution': {}
        }
        
        # Per-class analysis
        for class_label, class_info in class_similarities.items():
            class_paths = [p for p in paths_with_scores if p['class_label'] == class_label]
            
            analysis['per_class_analysis'][class_label] = {
                'num_paths': len(class_paths),
                'max_similarity': class_info['max_similarity'],
                'mean_similarity': class_info['mean_similarity'],
                'valid_paths': len([p for p in class_paths if p['path_length'] > 0]),
                'path_details': [
                    {
                        'path_index': p['path_index'],
                        'similarity': p['similarity_scores']['final_score'],
                        'length': p['path_length'],
                        'subactions': p['subaction_names'][:3]  # First 3 subactions for brevity
                    } for p in class_paths
                ]
            }
        
        # Overall statistics
        all_scores = [p['similarity_scores']['final_score'] for p in paths_with_scores if p['path_length'] > 0]
        if all_scores:
            analysis['overall_statistics'] = {
                'total_valid_paths': len(all_scores),
                'max_similarity': max(all_scores),
                'min_similarity': min(all_scores),
                'mean_similarity': sum(all_scores) / len(all_scores),
                'std_similarity': torch.tensor(all_scores).std().item() if len(all_scores) > 1 else 0.0
            }
        
        return analysis
    

    def forward(self, temporal_repr: torch.Tensor, 
                stream_type: str = 'subaction') -> torch.Tensor:
        """
        Forward pass of the graph matching module.
        
        Args:
            temporal_repr: Temporal representations [num_frames, embedding_dim]
            stream_type: Type of stream ('subaction', 'object', or 'class')
            
        Returns:
            Class confidence scores
        """
        # Select appropriate node embeddings (encoded) based on stream type
        if stream_type == 'subaction':
            node_embeddings = self.askg.subaction_embeddings
        elif stream_type == 'object':
            node_embeddings = self.askg.object_embeddings
        else:
            node_embeddings = self.askg.class_embeddings
        
        if node_embeddings is None:
            raise ValueError(f"No embeddings loaded for {stream_type} nodes")
        
        # Step 1: Node Alignment
        similarity_matrix, mapping_matrix = self.node_alignment(
            temporal_repr, node_embeddings
        )
        
        # Step 2: Build Interested Graph
        # interested_nodes = self.get_interested_nodes(
        #     mapping_matrix, similarity_matrix, node_embeddings
        # )

        interested_graph = self.build_interested_graph(mapping_matrix, similarity_matrix, node_embeddings)

        # Step 3: Graph Alignment
        ipdb.set_trace()

        graph_similarity = self.graph_alignment(temporal_repr, interested_graph)
        similarity_vec = []
        for key, items in graph_similarity['best_paths'].items():
            
        

        # Step 4: Integration
        # Calculate path-level confidence (average of top-k similarities)
        path_confidence = torch.mean(interested_graph['path_similarities'], dim=0)
        path_confidence = torch.mean(path_confidence)  # Average across all representations
        
        # Expand path_confidence to match graph_similarity dimensions
        path_confidence_expanded = path_confidence.expand_as(graph_similarity)
        
        # # Adaptive fusion of path and graph level similarities
        # total_similarity = self.adaptive_fusion(
        #     path_confidence_expanded.unsqueeze(0),
        #     graph_similarity.unsqueeze(0)
        # ).squeeze(0)
        
        # simple fuse
        total_similarity = path_confidence_expanded + graph_similarity

        return total_similarity


class ZeroShotASKG(nn.Module):
    """
    Zero-shot action recognition using ASKG and graph matching.
    Integrates with the existing VAE framework.
    """
    
    def __init__(self, askg: ASKG, vae_dict: Dict, embedding_dim: int = 96,
                 top_k: int = 5, num_classes: int = 60):
        super().__init__()
        self.askg = askg
        self.vae_dict = vae_dict
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.num_classes = num_classes
        
        # Graph matcher for different streams
        self.subaction_matcher = GraphMatcher(
            askg, embedding_dim, top_k, num_classes
        )
        
        # Stream fusion weights
        self.stream_weights = nn.Parameter(torch.ones(1))
        
    def forward(self, skeleton_data: torch.Tensor, 
                num_reps: int = 4) -> torch.Tensor:
        """
        Forward pass for zero-shot classification.
        
        Args:
            skeleton_data: Input skeleton sequence [batch_size, embedding_size, frames]
            num_reps: Number of frames to sample for temporal representation
            
        Returns:
            Class prediction scores
        """
        batch_size = skeleton_data.size(0)

        # Sample frames for temporal representation
        indices = torch.linspace(0, skeleton_data.shape[2] - 1, num_reps).long()  # 改为 rep_mode可选
        rep_segs = skeleton_data[:, :, indices]  # [batch_size, embedding_size, num_frames]
        
        # Encode each frame using VAE encoder
        sequence_encoder = self.vae_dict['sub_act']['sequence_encoder']
        sequence_encoder.eval()
        
        batch_predictions = []
        
        for b in range(batch_size):
            frame_representations = [] 
            
            # Encode each frame
            for f in range(num_reps):
                frame_data = rep_segs[b, :, f].unsqueeze(0)  # [1, channels]
                mu, logvar = sequence_encoder(frame_data)
                frame_representations.append(mu.squeeze(0))  # [embedding_dim]  
            
            temporal_rep = torch.stack(frame_representations, dim=0)  # [num_frames, embedding_dim]
            
            # Graph matching for sub-action stream
            subaction_scores = self.subaction_matcher(temporal_rep, 'subaction')
            
            # For now, only use sub-action stream
            # In future, can add object and class streams and fuse them
            final_scores = subaction_scores
            
            batch_predictions.append(final_scores)
        
        return torch.stack(batch_predictions, dim=0)  # [batch_size, num_classes]


# Usage example and utility functions
def create_sample_askg_mapping() -> Dict:
    """Create a sample ASKG mapping for demonstration."""
    return {
        'cls2obj': {
            'drink water': ['bottle', 'cup', 'glass'],
            'eat meal': ['spoon', 'fork', 'plate', 'bowl'],
            'brush teeth': ['toothbrush', 'toothpaste'],
            'read': ['book', 'paper', 'magazine'],
            'write': ['pen', 'pencil', 'paper']
        },
        'cls2sa': {
            'drink water': ['reach for container', 'grasp container', 'lift to mouth', 'tilt and drink'],
            'eat meal': ['pick up utensil', 'reach for food', 'bring to mouth', 'chew'],
            'brush teeth': ['apply toothpaste', 'start brushing', 'brush upper teeth', 'brush lower teeth'],
            'read': ['open book', 'focus on text', 'turn page'],
            'write': ['hold pen', 'position on paper', 'make strokes', 'lift pen']
        }
    }


def load_askg_embeddings(askg: ASKG, sa_emb, text_encoder, device: torch.device) -> ASKG:
    """
    Load pre-trained embeddings for ASKG nodes.
    This is a placeholder - in practice, you'd load from saved embeddings.
    """
    node_info = askg.get_node_info()
    embedding_dim = askg.embedding_dim
    
    # Create random embeddings as placeholder
    class_emb = torch.randn(node_info['num_classes'], embedding_dim).to(device)
    object_emb = torch.randn(node_info['num_objects'], embedding_dim).to(device)
    
    # load from sub-action semantic embeddings
    sa_emb = sa_emb.to(torch.float32)
    t_tmu, t_tlv = text_encoder(sa_emb)
    t_z = reparameterize(t_tmu, t_tlv)
    subaction_emb = t_z.to(device)  # torch.Size([154, 96])
    
    askg.load_embeddings(class_emb, object_emb, subaction_emb)
    return askg


def graph_match_vae(vae_dict: Dict, num_unseen_class, askg_mapping: Dict, sa_emb, semantic_latent_size, device) -> ZeroShotASKG:
    """
    Integrate ASKG with existing VAE framework.
    
    Args:
        vae_dict: Dictionary containing VAE models
        askg_mapping: ASKG mapping dictionary
        device: Torch device
        
    Returns:
        Initialized ZeroShotASKG model
    """
    # Initialize action semantic knowledge graph
    askg = ASKG(askg_mapping, embedding_dim=semantic_latent_size)
    
    # load the encoded semantic embeddings
    text_encoder = vae_dict['sub_act']['text_encoder']
    text_encoder.eval()
    askg = load_askg_embeddings(askg, sa_emb, text_encoder, device)
    
    # Initialize the zero-shot graph matching model
    zs_askg = ZeroShotASKG(
        askg=askg,
        vae_dict=vae_dict,
        embedding_dim=96,  # Should match VAE encoder output
        top_k=5,
        num_classes=num_unseen_class
    ).to(device)
    
    return zs_askg

'''
# In the main function, after VAE training:

# load from file:
with open('ASKG/data/ntu/askg_mappings.json', 'r') as f:
    askg_mapping = json.load(f)

# 2. Integrate with existing framework
zs_askg_model = graph_match_vae(vae_dict, askg_mapping, device)

# 3. Use for zero-shot classification
zs_askg_model.eval()
with torch.no_grad():
    for batch_data, targets in test_loader:
        batch_data = batch_data.to(device)
        
        # Get predictions using graph matching
        predictions = zs_askg_model(batch_data, num_frames=4)
        
        # Calculate accuracy
        predicted_classes = torch.argmax(predictions, dim=1)
        # ... rest of evaluation code
'''