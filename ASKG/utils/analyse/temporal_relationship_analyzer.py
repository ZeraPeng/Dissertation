import yaml
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

# Load ASKG data
def load_askg_data(file_path):
    """Load ASKG data from YAML file"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

# Extract temporal relationships from act_rel_triples
def extract_temporal_relations(askg_data):
    """Extract temporal relationships between subactions"""
    # Dictionary to store relationships
    relationships = []
    
    # 首先尝试从act_rel_triples中提取关系
    for action_name, action_info in askg_data.items():
        if 'act_rel_triples' in action_info and action_info['act_rel_triples']:
            for rel_triple in action_info['act_rel_triples']:
                # 尝试从字符串中提取关系
                triple_str = str(rel_triple)
                
                # 使用更灵活的正则表达式来匹配三元组
                match = re.search(r'<\s*([^,<>]+)\s*,\s*([^,<>]+)\s*,\s*([^,<>]+)\s*>', triple_str)
                
                if match:
                    subact1, relation, subact2 = match.groups()
                    subact1 = subact1.strip()
                    relation = relation.strip()
                    subact2 = subact2.strip()
                    
                    # 简化检查 - 只要包含"precedes"或"follows"即可
                    if "precedes" in relation.lower():
                        relationships.append((subact1, subact2, "precedes", action_name))
                    elif "follows" in relation.lower():
                        relationships.append((subact2, subact1, "precedes", action_name))
    
    # 如果没有找到任何关系，使用子动作列表的顺序生成关系
    if not relationships:
        print("No temporal relationships found in act_rel_triples. Using subaction order instead.")
        for action_name, action_info in askg_data.items():
            if 'sub_act_li' in action_info and action_info['sub_act_li']:
                subacts = action_info['sub_act_li']
                
                # 只有当列表中有至少两个子动作时才创建关系
                if len(subacts) >= 2:
                    for i in range(len(subacts) - 1):
                        subact1 = subacts[i]
                        subact2 = subacts[i + 1]
                        # 假设列表中的顺序代表时间顺序
                        relationships.append((subact1, subact2, "precedes", action_name))
    
    print(f"Total temporal relationships extracted/generated: {len(relationships)}")
    return relationships

# Build directed graph from temporal relationships
def build_temporal_graph(relationships):
    """Build directed graph from temporal relationships"""
    G = nx.DiGraph()
    
    # Add edges for each relationship
    for subact1, subact2, relation, action_name in relationships:
        if not G.has_edge(subact1, subact2):
            G.add_edge(subact1, subact2, actions=[action_name], weight=1)
        else:
            G[subact1][subact2]['actions'].append(action_name)
            G[subact1][subact2]['weight'] += 1
    
    return G

# Visualize temporal graph
def visualize_temporal_graph(G, title="Subaction Temporal Graph", min_edge_weight=2):
    """Visualize temporal relationships as a directed graph"""
    # Filter edges by weight if needed
    if min_edge_weight > 1:
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_edge_weight]
        G_filtered = G.copy()
        G_filtered.remove_edges_from(edges_to_remove)
    else:
        G_filtered = G
    
    # Filter isolated nodes
    G_filtered = G_filtered.subgraph([n for n in G_filtered.nodes() if G_filtered.degree(n) > 0])
    
    plt.figure(figsize=(16, 12))
    
    # Compute node sizes based on degree
    node_sizes = {n: 100 + 50 * G_filtered.degree(n) for n in G_filtered.nodes()}
    
    # Use a hierarchical layout for better visualization of sequences
    pos = nx.spring_layout(G_filtered, k=0.15, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_filtered, pos, 
                         node_size=[node_sizes[n] for n in G_filtered.nodes()],
                         node_color='lightblue', 
                         alpha=0.8)
    
    # Draw edges with width proportional to weight
    edges = G_filtered.edges(data=True)
    edge_widths = [d['weight'] * 0.5 for _, _, d in edges]
    
    nx.draw_networkx_edges(G_filtered, pos, 
                         width=edge_widths,
                         alpha=0.6, 
                         edge_color='gray',
                         arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G_filtered, pos, font_size=10, font_family='sans-serif')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

# Identify common subaction sequences
def identify_common_sequences(relationships, min_support=2):
    """Identify common subaction sequences"""
    # Dictionary to count sequence occurrences
    sequence_counter = Counter()
    
    # Group relationships by action
    action_relations = defaultdict(list)
    for subact1, subact2, relation, action_name in relationships:
        action_relations[action_name].append((subact1, subact2))
    
    # Find all paths of length 2
    for action_name, relations in action_relations.items():
        # Build a directed graph for this action
        G = nx.DiGraph()
        for subact1, subact2 in relations:
            G.add_edge(subact1, subact2)
        
        # Find all simple paths of length 2
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    paths = list(nx.all_simple_paths(G, source, target, cutoff=2))
                    for path in paths:
                        if len(path) >= 2:  # Path has at least 2 nodes (1 edge)
                            sequence_counter[tuple(path)] += 1
    
    # Filter sequences by minimum support
    common_sequences = {seq: count for seq, count in sequence_counter.items() if count >= min_support}
    
    return common_sequences

# Create transition matrix for subactions
def create_transition_matrix(relationships):
    """Create transition matrix for subactions"""
    # Extract all unique subactions
    all_subactions = set()
    for subact1, subact2, _, _ in relationships:
        all_subactions.add(subact1)
        all_subactions.add(subact2)
    
    all_subactions = sorted(list(all_subactions))
    subact_to_idx = {subact: i for i, subact in enumerate(all_subactions)}
    
    # Initialize transition matrix
    n = len(all_subactions)
    transition_matrix = np.zeros((n, n))
    
    # Fill transition matrix
    for subact1, subact2, _, _ in relationships:
        i = subact_to_idx[subact1]
        j = subact_to_idx[subact2]
        transition_matrix[i, j] += 1
    
    return transition_matrix, all_subactions

# Visualize transition matrix
def visualize_transition_matrix(matrix, subactions, title="Subaction Transition Matrix"):
    """Visualize transition matrix as a heatmap"""
    # 检查矩阵是否为空
    if matrix.size == 0 or len(subactions) == 0:
        print("Warning: Empty matrix or no subactions to visualize")
        # 创建一个空图形并返回
        fig = plt.figure(figsize=(10, 8))
        plt.title("No data to visualize")
        return fig
    
    plt.figure(figsize=(14, 12))
    
    # Create mask for zero values
    mask = matrix == 0
    
    # Create custom colormap
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap=cmap, mask=mask,
                xticklabels=subactions, yticklabels=subactions,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title(title, fontsize=16)
    plt.xlabel('To Subaction', fontsize=14)
    plt.ylabel('From Subaction', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt.gcf()

# Analyze subaction positions in sequences
def analyze_subaction_positions(askg_data):
    """Analyze positions of subactions in action sequences"""
    position_data = []
    
    for action_name, action_info in askg_data.items():
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            subactions = action_info['sub_act_li']
            total_steps = len(subactions)
            
            for pos, subact in enumerate(subactions):
                position_data.append({
                    'action': action_name,
                    'subaction': subact,
                    'position': pos,
                    'relative_position': pos / max(1, total_steps - 1) if total_steps > 1 else 0.5,
                    'total_steps': total_steps
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(position_data)
    
    return df

# Visualize subaction positions
def visualize_subaction_positions(position_df, min_occurrences=3):
    """Visualize the typical positions of subactions in sequences"""
    # Count occurrences of each subaction
    subact_counts = position_df['subaction'].value_counts()
    
    # Filter subactions by minimum occurrences
    common_subactions = subact_counts[subact_counts >= min_occurrences].index.tolist()
    filtered_df = position_df[position_df['subaction'].isin(common_subactions)]
    
    # Group by subaction and calculate position statistics
    subact_positions = filtered_df.groupby('subaction')['relative_position'].agg(['mean', 'std', 'count'])
    subact_positions = subact_positions.sort_values('mean')
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bars for each subaction
    y_pos = np.arange(len(subact_positions))
    
    # Plot mean positions
    plt.barh(y_pos, subact_positions['mean'], xerr=subact_positions['std'],
             height=0.6, alpha=0.7, color='skyblue', capsize=5)
    
    # Add labels
    plt.yticks(y_pos, subact_positions.index)
    plt.xlabel('Relative Position in Action Sequence (0 = start, 1 = end)')
    plt.title('Typical Positions of Subactions in Action Sequences', fontsize=16)
    
    # Add a grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add count as text
    for i, (_, row) in enumerate(subact_positions.iterrows()):
        plt.text(1.01, i, f'n = {int(row["count"])}', va='center')
    
    plt.tight_layout()
    
    return plt.gcf()

# Main function
def main(askg_file=None, output_dir=None):
    """Main function"""
    # 如果没有提供参数，则使用默认参数或解析命令行参数
    if askg_file is None or output_dir is None:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Analyze temporal relationships in ASKG')
        parser.add_argument('--askg', type=str, default='classes_ASKG_ntu_checked.yml',
                          help='Path to ASKG data file')
        parser.add_argument('--output', type=str, default='temporal_output',
                          help='Output directory')
        args = parser.parse_args()
        
        # 如果没有提供参数，则使用命令行参数
        if askg_file is None:
            askg_file = args.askg
        if output_dir is None:
            output_dir = args.output
    
    # Load ASKG data
    askg_data = load_askg_data('classes_ASKG_ntu_checked.yml')
    
    # Extract temporal relationships
    relationships = extract_temporal_relations(askg_data)
    print(f"Extracted {len(relationships)} temporal relationships from ASKG data")
    
    # Build and visualize temporal graph
    G = build_temporal_graph(relationships)
    print(f"Temporal graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    graph_viz = visualize_temporal_graph(G, "Subaction Temporal Relationships (All)")
    graph_viz.savefig(os.path.join(output_dir, 'temporal_graph_all.png'))
    
    # Visualize with minimum edge weight
    min_weight = 2
    graph_viz_filtered = visualize_temporal_graph(G, f"Subaction Temporal Relationships (Weight ≥ {min_weight})", min_weight)
    graph_viz_filtered.savefig(os.path.join(output_dir, f'temporal_graph_weight_{min_weight}.png'))
    
    # Identify common sequences
    common_sequences = identify_common_sequences(relationships, min_support=3)
    print("\nCommon subaction sequences (support ≥ 3):")
    for seq, count in sorted(common_sequences.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {' → '.join(seq)}: {count}")
    
    # Create and visualize transition matrix
    transition_matrix, subactions = create_transition_matrix(relationships)
    
    # Filter matrix to only show common subactions (for better visualization)
    common_threshold = 3
    row_sums = transition_matrix.sum(axis=1)
    col_sums = transition_matrix.sum(axis=0)
    common_indices = np.where((row_sums >= common_threshold) | (col_sums >= common_threshold))[0]
    
    filtered_matrix = transition_matrix[common_indices][:, common_indices]
    filtered_subactions = [subactions[i] for i in common_indices]
    
    if common_indices.size == 0:
        print("Warning: No common subactions found with threshold", common_threshold)
        # 创建一个空图形
        matrix_viz = plt.figure(figsize=(10, 8))
        plt.title("No common subactions found")
    else:
        filtered_matrix = transition_matrix[common_indices][:, common_indices]
        filtered_subactions = [subactions[i] for i in common_indices]
    
    matrix_viz = visualize_transition_matrix(filtered_matrix, filtered_subactions)
    
    matrix_viz = visualize_transition_matrix(filtered_matrix, filtered_subactions)
    matrix_viz.savefig(os.path.join(output_dir, 'transition_matrix.png'))
    
    # Analyze subaction positions
    position_df = analyze_subaction_positions(askg_data)
    position_viz = visualize_subaction_positions(position_df)
    position_viz.savefig(os.path.join(output_dir, 'subaction_positions.png'))
    
    print("\nAnalysis complete. Visualizations saved to:")
    print(f"  - {os.path.join(output_dir, 'temporal_graph_all.png')}")
    print(f"  - {os.path.join(output_dir, f'temporal_graph_weight_{min_weight}.png')}")
    print(f"  - {os.path.join(output_dir, 'transition_matrix.png')}")
    print(f"  - {os.path.join(output_dir, 'subaction_positions.png')}")

if __name__ == "__main__":
    main()