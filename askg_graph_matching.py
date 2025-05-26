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
    
    # def get_interested_nodes(self, mapping_matrix: torch.Tensor, 
    #                           similarity_matrix: torch.Tensor,
    #                           node_embeddings: torch.Tensor) -> Dict:
    #     """
    #     Build interested subgraph from top-k nodes.
        
    #     Args:
    #         mapping_matrix: Mapping matrix from node alignment
    #         similarity_matrix: Similarity scores between representations and nodes
    #         node_embeddings: Node embeddings
            
    #     Returns:
    #         Dictionary containing interested graph information
    #     """
    #     # Get interested node set
    #     interested_nodes = torch.unique(mapping_matrix.flatten())   # tensor([  9,  14,  15,  25,  42,  49,  67,  84, 107, 112, 115, 129]
        
    #     # Extract subgraph embeddings
    #     subgraph_node_embeddings = node_embeddings[interested_nodes]     # torch.Size([12, 96])
        
    #     # Generate path-level similarity vectors (scores)
    #     path_similarities = []
    #     for i in range(mapping_matrix.size(0)): # 4
    #         top_k_nodes = mapping_matrix[i]
    #         path_sim = similarity_matrix[i, top_k_nodes]
    #         path_similarities.append(path_sim)
        
    #     path_similarities = torch.stack(path_similarities, dim=0)
        
    #     return {
    #         'interested_nodes': interested_nodes,
    #         'subgraph_embeddings': subgraph_node_embeddings,
    #         'path_similarities': path_similarities
    #     }
    
    # def build_interested_graphs(self, interested_nodes_info: Dict) -> Dict:
    #     """
    #     Build interested subgraph containing all paths that include the interested nodes.
        
    #     Args:
    #         interested_nodes_info: Dictionary containing interested nodes information
            
    #     Returns:
    #         Dictionary containing the interested subgraph with paths and similarity vectors
    #     """
    #     ipdb.set_trace()
    #     interested_nodes = interested_nodes_info['interested_nodes']  # tensor of node indices
    #     path_similarities = interested_nodes_info['path_similarities']  # similarities for each representation
    #     subgraph_embeddings = interested_nodes_info['subgraph_embeddings']
        
    #     # Get adjacency matrices for building paths
    #     adj_matrices = self.askg.get_adjacency_matrices()
    #     precedes_adj = adj_matrices['precedes']  # [num_subactions, num_subactions]
        
    #     # Convert interested_nodes to set for efficient lookup
    #     interested_nodes_set = set(interested_nodes.cpu().numpy())
        
    #     # Find all paths in the knowledge graph that contain interested nodes
    #     all_paths = []
    #     path_node_similarities = []  # Store similarity vectors for each path
        

    #     num_subactions = precedes_adj.size(0)
        
    #     # For each pair of interested nodes, find paths between them
    #     for i, start_node in enumerate(interested_nodes):
    #         start_idx = start_node.item()
            
    #         # DFS to find all paths starting from this node
    #         visited_paths = self._find_paths_from_node(
    #             start_idx, precedes_adj, interested_nodes_set, max_depth=5
    #         )
            
    #         for path in visited_paths:
    #             if len(path) >= 2:  # Only consider paths with at least 2 nodes
    #                 all_paths.append(path)
                    
    #                 # Create similarity vector for this path
    #                 path_sim_vector = []
    #                 for node_idx in path:
    #                     # Find which representation this node belongs to and get its similarity
    #                     node_tensor = torch.tensor(node_idx, device=interested_nodes.device)
    #                     if node_tensor in interested_nodes:
    #                         # Find the position of this node in interested_nodes
    #                         pos = (interested_nodes == node_tensor).nonzero(as_tuple=True)[0]
    #                         if len(pos) > 0:
    #                             # Get similarity from the corresponding representation
    #                             repr_idx = pos[0] % path_similarities.size(0)  # Handle case where node appears multiple times
    #                             topk_idx = pos[0] % path_similarities.size(1)
    #                             sim_score = path_similarities[repr_idx, topk_idx]
    #                             path_sim_vector.append(sim_score)
    #                         else:
    #                             path_sim_vector.append(torch.tensor(0.0, device=interested_nodes.device))
    #                     else:
    #                         # Node not in interested nodes, assign average similarity
    #                         path_sim_vector.append(torch.mean(path_similarities))
                    
    #                 if path_sim_vector:
    #                     path_node_similarities.append(torch.stack(path_sim_vector))
        
    #     # Remove duplicate paths
    #     unique_paths = []
    #     unique_path_similarities = []
    #     seen_paths = set()
        
    #     for i, path in enumerate(all_paths):
    #         path_tuple = tuple(sorted(path))  # Sort for consistent comparison
    #         if path_tuple not in seen_paths:
    #             seen_paths.add(path_tuple)
    #             unique_paths.append(path)
    #             if i < len(path_node_similarities):
    #                 unique_path_similarities.append(path_node_similarities[i])
        
    #     # Create subgraph adjacency matrix for interested nodes only
    #     num_interested = len(interested_nodes)
    #     subgraph_adj = torch.zeros(num_interested, num_interested, device=interested_nodes.device)
        
    #     # Map global node indices to local subgraph indices
    #     global_to_local = {node_idx.item(): i for i, node_idx in enumerate(interested_nodes)}
        
    #     # Fill adjacency matrix for subgraph
    #     for i, node_i in enumerate(interested_nodes):
    #         for j, node_j in enumerate(interested_nodes):
    #             if i != j and precedes_adj[node_i.item(), node_j.item()] > 0:
    #                 subgraph_adj[i, j] = 1
        
    #     return {
    #         'interested_nodes': interested_nodes,
    #         'subgraph_embeddings': subgraph_embeddings,
    #         'path_similarities': path_similarities,
    #         'all_paths': unique_paths,
    #         'path_node_similarities': unique_path_similarities,
    #         'subgraph_adjacency': subgraph_adj,
    #         'global_to_local_mapping': global_to_local,
    #         'num_paths': len(unique_paths)
    #     }

    # def _find_paths_from_node(self, start_node: int, adj_matrix: torch.Tensor, 
    #                         interested_nodes_set: set, max_depth: int = 5) -> List[List[int]]:
    #     """
    #     Find all paths starting from a given node using DFS.
        
    #     Args:
    #         start_node: Starting node index
    #         adj_matrix: Adjacency matrix for the graph
    #         interested_nodes_set: Set of interested node indices
    #         max_depth: Maximum path depth to explore
            
    #     Returns:
    #         List of paths, where each path is a list of node indices
    #     """
    #     paths = []
    
    #     def dfs(current_node: int, current_path: List[int], visited: set, depth: int):
    #         if depth > max_depth:
    #             return
                
    #         # Add current path if it contains interested nodes and has reasonable length
    #         if len(current_path) >= 1 and any(node in interested_nodes_set for node in current_path):
    #             paths.append(current_path.copy())
            
    #         # Explore neighbors
    #         neighbors = torch.nonzero(adj_matrix[current_node], as_tuple=True)[0]
    #         for neighbor in neighbors:
    #             neighbor_idx = neighbor.item()
    #             if neighbor_idx not in visited:  # Avoid cycles
    #                 visited.add(neighbor_idx)
    #                 current_path.append(neighbor_idx)
    #                 dfs(neighbor_idx, current_path, visited, depth + 1)
    #                 current_path.pop()
    #                 visited.remove(neighbor_idx)
    
    #     # Start DFS from the starting node
    #     visited = {start_node}
    #     dfs(start_node, [start_node], visited, 0)
        
    #     return paths

    def graph_alignment(self, temporal_repr: torch.Tensor, 
                       interested_graph: Dict) -> torch.Tensor:
        """
        Perform graph alignment between sample graph and interested subgraph.
        
        Args:
            temporal_repr: Temporal representations
            interested_graph: Dictionary containing interested graph info
            
        Returns:
            Graph-level similarity vector
        """
        # Build sample graph from temporal representations
        sample_graph = temporal_repr.unsqueeze(0)  # Add batch dimension    # torch.Size([1, 4, 96])
        
        # Get interested subgraph embeddings
        subgraph_emb = interested_graph['subgraph_embeddings'].unsqueeze(0)     # torch.Size([1, 12, 96])
        
        # Apply graph attention for alignment
        aligned_features = self.graph_attention(
            query=sample_graph,
            key=subgraph_emb,
            value=subgraph_emb
        )       # torch.Size([1, 4, 96])
        
        # Calculate graph-level similarity scores for each class
        graph_similarity = torch.zeros(self.num_classes, device=temporal_repr.device)
        
        # Simple aggregation - can be made more sophisticated
        aggregated_features = torch.mean(aligned_features, dim=1)  # [1, embedding_dim]
        class_logits = self.classifier(aggregated_features)  # [1, num_classes]
        graph_similarity = torch.softmax(class_logits.squeeze(0), dim=0)    # tensor([0.2072, 0.2060, 0.1764, 0.2015, 0.2089], device='cuda:0')
        
        return graph_similarity
    
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

        interested_graph = self.build_interested_graphs(mapping_matrix)

        # Step 3: Graph Alignment
        graph_similarity = self.graph_alignment(temporal_repr, interested_graph)
        
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