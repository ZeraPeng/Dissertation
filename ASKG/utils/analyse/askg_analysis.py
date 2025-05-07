import numpy as np
import matplotlib.pyplot as plt
import yaml
import re
from collections import Counter
from sklearn.manifold import TSNE
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from itertools import combinations

# Load ASKG data
def load_askg_data(file_path):
    """Load ASKG data from YAML file"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

# Add body part definitions
def get_part_names(body_part=4):
    if body_part == 4:
        return ["head", "upper limbs", "hip", "lower limbs"]
    # Other options...

# Body part keywords
BODY_PART_KEYWORDS = {
    "head": ["head", "face", "ear", "nose", "mouth", "neck", "eyes", "hair", "glasses"],
    "upper limbs": ["hand", "arm", "finger", "palm", "wrist", "elbow", "shoulder", "thumb"],
    "hip": ["hip", "waist", "torso", "chest", "back", "stomach"],
    "lower limbs": ["leg", "foot", "knee", "ankle", "toe", "feet", "thigh"]
}

# Identify if an object is a body part
def identify_body_parts(obj_list):
    """Identify body parts from a list of objects"""
    body_parts = set()
    
    for obj in obj_list:
        obj_name = obj
        if isinstance(obj, dict) and 'object' in obj:
            obj_name = obj['object']
            
        # Check if object is a body part
        for part, keywords in BODY_PART_KEYWORDS.items():
            if any(keyword in obj_name.lower() for keyword in keywords):
                body_parts.add(part)
                break
    
    return list(body_parts)

# Objects to exclude
EXCLUDED_OBJECTS = ["air", "sound", "environment", "space", "distance", "surface", "balance"]
# Extract vocabularies from ASKG

# Extract vocabularies from ASKG
def extract_vocabularies(askg_data):
    """Extract object and subaction vocabularies from ASKG data"""
    objects = []
    subactions = []
    body_parts = set()
    
    for action_label, action_info in askg_data.items():
        # Extract objects
        if 'obj_li' in action_info and action_info['obj_li']:
            for obj in action_info['obj_li']:
                if isinstance(obj, dict):
                    # Handle case where obj is a dictionary with 'object' key
                    if 'object' in obj:
                        obj_name = obj['object']
                        
                        # Check if object is in excluded list
                        if obj_name.lower() not in EXCLUDED_OBJECTS:
                            # Check if object is a body part
                            is_body_part = False
                            for part, keywords in BODY_PART_KEYWORDS.items():
                                if any(keyword in obj_name.lower() for keyword in keywords):
                                    body_parts.add(part)
                                    is_body_part = True
                                    break
                            
                            # If not a body part, add to objects list
                            if not is_body_part:
                                objects.append(obj_name)
                else:
                    obj_name = obj
                    
                    # Check if object is in excluded list
                    if obj_name.lower() not in EXCLUDED_OBJECTS:
                        # Check if object is a body part
                        is_body_part = False
                        for part, keywords in BODY_PART_KEYWORDS.items():
                            if any(keyword in obj_name.lower() for keyword in keywords):
                                body_parts.add(part)
                                is_body_part = True
                                break
                        
                        # If not a body part, add to objects list
                        if not is_body_part:
                            objects.append(obj_name)
        
        # Extract subactions (unchanged)
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                subactions.append(subact)
    
    # Count occurrences
    object_counter = Counter(objects)
    subaction_counter = Counter(subactions)
    body_part_counter = Counter(list(body_parts))
    
    return object_counter, subaction_counter, body_part_counter

# Create co-occurrence matrix
def create_cooccurrence_matrix(askg_data, top_k_objects=5, top_k_subactions=5):
    """Create a co-occurrence matrix between top-K objects and subactions"""
    try:
        # Try new version of extract_vocabularies
        object_counter, subaction_counter, _ = extract_vocabularies(askg_data)
    except ValueError:
        # Compatible with old version
        object_counter, subaction_counter = extract_vocabularies(askg_data)
    
    # Get top-K objects and subactions
    top_objects = [obj for obj, _ in object_counter.most_common(top_k_objects)]
    top_subactions = [act for act, _ in subaction_counter.most_common(top_k_subactions)]
    
    # Initialize co-occurrence matrix
    cooccurrence_matrix = np.zeros((len(top_subactions), len(top_objects)))
    
    # Fill matrix
    for action_label, action_info in askg_data.items():
        action_objects = []
        action_subactions = []
        
        # Extract objects for this action
        if 'obj_li' in action_info and action_info['obj_li']:
            for obj in action_info['obj_li']:
                if isinstance(obj, dict):
                    # Handle case where obj is a dictionary with 'object' key
                    if 'object' in obj:
                        obj_name = obj['object']
                        if obj_name in top_objects:
                            action_objects.append(obj_name)
                else:
                    if obj in top_objects:
                        action_objects.append(obj)
        
        # Extract subactions for this action
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                if subact in top_subactions:
                    action_subactions.append(subact)
        
        # Update co-occurrence matrix
        for subact in action_subactions:
            for obj in action_objects:
                subact_idx = top_subactions.index(subact)
                obj_idx = top_objects.index(obj)
                cooccurrence_matrix[subact_idx, obj_idx] += 1
    
    return cooccurrence_matrix, top_subactions, top_objects

# Create body part-subaction co-occurrence matrix
def create_body_part_subaction_matrix(askg_data, top_k_body_parts=4, top_k_subactions=20):
    """Create a co-occurrence matrix between body parts and subactions"""
    # Get vocabulary for body parts and subactions
    _, subaction_counter, body_part_counter = extract_vocabularies(askg_data)
    # Visualize body part-subaction co-occurrence matrix
def plot_body_part_subaction_matrix(matrix, row_labels, col_labels, title="Body Part-Subaction Co-occurrence Matrix", output_file=None):
    """Plot a labeled body part-subaction co-occurrence matrix"""
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap (similar to image 3)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    
    # Convert matrix to integer type
    matrix = np.array(matrix).astype(int)
    
    # Plot heatmap with annotations
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Body Parts', fontsize=14)
    plt.ylabel('Subactions', fontsize=14)
    plt.tight_layout()
    
    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return plt.gcf()
    # Get top-K body parts and subactions
    body_parts = get_part_names(4)  # Use 4-part body division
    top_subactions = [act for act, _ in subaction_counter.most_common(top_k_subactions)]
    
    # Initialize co-occurrence matrix
    cooccurrence_matrix = np.zeros((len(top_subactions), len(body_parts)))
    
    # Fill matrix
    for action_label, action_info in askg_data.items():
        action_body_parts = []
        action_subactions = []
        
        # Extract objects for this action and identify body parts
        if 'obj_li' in action_info and action_info['obj_li']:
            action_body_parts = identify_body_parts(action_info['obj_li'])
        
        # Extract subactions for this action
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                if subact in top_subactions:
                    action_subactions.append(subact)
        
        # Update co-occurrence matrix
        for subact in action_subactions:
            for body_part in action_body_parts:
                if body_part in body_parts:
                    subact_idx = top_subactions.index(subact)
                    body_part_idx = body_parts.index(body_part)
                    cooccurrence_matrix[subact_idx, body_part_idx] += 1
    
    return cooccurrence_matrix, top_subactions, body_parts

# Visualize co-occurrence matrix
def plot_cooccurrence_matrix(matrix, row_labels, col_labels, title="Co-occurrence Matrix", output_file=None):
    """Plot a labeled co-occurrence matrix"""
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap (similar to image 3)
    cmap = sns.color_palette("YlGn", as_cmap=True)
    
    # Convert matrix to integer type
    matrix = np.array(matrix).astype(int)
    
    # Now it's safe to use fmt="d"
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Object Classes', fontsize=14)
    plt.ylabel('Verb Classes', fontsize=14)
    plt.tight_layout()
    
    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return plt.gcf()

# Visualize body part-subaction co-occurrence matrix
def plot_body_part_subaction_matrix(matrix, row_labels, col_labels, title="Body Part-Subaction Co-occurrence Matrix", output_file=None):
    """Plot a labeled body part-subaction co-occurrence matrix"""
    plt.figure(figsize=(12, 10))
    
    # Create custom colormap (similar to image 3)
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    
    # Convert matrix to integer type
    matrix = np.array(matrix).astype(int)
    
    # Plot heatmap with annotations
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Body Parts', fontsize=14)
    plt.ylabel('Subactions', fontsize=14)
    plt.tight_layout()
    
    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return plt.gcf()

# Analyze subaction temporal relationships
def analyze_subaction_temporal_relations(askg_data):
    """Analyze temporal relationships between subactions"""
    # Extract all unique subactions
    all_subactions = set()
    for action_label, action_info in askg_data.items():
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            all_subactions.update(action_info['sub_act_li'])
    
    # Create a directed graph for temporal relationships
    G = nx.DiGraph()
    
    # Add nodes for all subactions
    for subact in all_subactions:
        G.add_node(subact)
    
    # Add edges based on temporal relationships
    for action_label, action_info in askg_data.items():
        if 'act_rel_triples' in action_info and action_info['act_rel_triples']:
            for rel_triple in action_info['act_rel_triples']:
                # Parse the relationship triple
                triple_str = str(rel_triple)
                match = re.search(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', triple_str)
                
                if match:
                    subact1, relation, subact2 = match.groups()
                    subact1 = subact1.strip()
                    relation = relation.strip()
                    subact2 = subact2.strip()
                    
                    # Check if relation indicates temporal ordering
                    temporal_relations = ['precedes', 'follows', 'before', 'after', 'comes before']
                    if any(rel in relation for rel in temporal_relations):
                        if 'precedes' in relation or 'before' in relation or 'comes before' in relation:
                            G.add_edge(subact1, subact2, relation=relation)
                        elif 'follows' in relation or 'after' in relation:
                            G.add_edge(subact2, subact1, relation=relation)
    
    return G, all_subactions

# Visualize subaction temporal relationships
def plot_subaction_temporal_network(G, title="Subaction Temporal Relationships"):
    """Visualize temporal relationships between subactions as a network"""
    plt.figure(figsize=(14, 10))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', 
                          arrows=True, arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

# Function to create vocabulary from existing ASKG
def create_vocabulary_from_existing_askg(askg_data):
    """Create object, subaction, and body part vocabularies from existing ASKG data"""
    object_counter, subaction_counter, body_part_counter = extract_vocabularies(askg_data)
    
    # Create vocabularies
    object_vocabulary = list(object_counter.keys())
    subaction_vocabulary = list(subaction_counter.keys())
    body_part_vocabulary = list(body_part_counter.keys())
    
    return object_vocabulary, subaction_vocabulary, body_part_vocabulary

# Generate t-SNE visualization (similar to image 1)
def generate_tsne_visualization(askg_data, feature_type='skeleton'):
    """Generate t-SNE visualization of action categories"""
    # For demonstration, we'll create random feature vectors for each action
    # In reality, these would come from actual feature embeddings
    
    labels = list(askg_data.keys())
    n_samples = len(labels)
    
    # Create random features (in real application, these would be actual embeddings)
    if feature_type == 'skeleton':
        features = np.random.rand(n_samples, 100)  # 100-dim features for skeleton space
    else:  # semantic
        features = np.random.rand(n_samples, 100)  # 100-dim features for semantic space
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    tsne_results = tsne.fit_transform(features)
    
    # Visualize results
    plt.figure(figsize=(10, 8))
    
    # Create color map (one color per category)
    cmap = plt.cm.get_cmap('tab20', n_samples)
    
    # Plot points
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=range(n_samples), 
                          cmap=cmap, s=100, alpha=0.8)
    
    # Add title
    if feature_type == 'skeleton':
        plt.title('t-SNE visualization of skeleton space', fontsize=16)
    else:
        plt.title('t-SNE visualization of semantic space', fontsize=16)
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

# Visualization for subaction temporal patterns
def visualize_subaction_patterns(askg_data, action_name):
    """Visualize the temporal pattern of subactions for a specific action"""
    action_info = askg_data.get(action_name)
    
    if not action_info:
        return None, f"Action '{action_name}' not found in the data."
    
    if 'sub_act_li' not in action_info or not action_info['sub_act_li']:
        return None, f"No subactions found for action '{action_name}'."
    
    # Get subactions
    subactions = action_info['sub_act_li']
    
    # Create a simple timeline visualization
    plt.figure(figsize=(12, 6))
    
    # Plot subactions as a sequence
    for i, subact in enumerate(subactions):
        plt.plot([i, i+1], [1, 1], 'bo-', linewidth=2, markersize=10)
        plt.text(i+0.5, 1.1, subact, ha='center', fontsize=12)
    
    # Connect subactions with arrows to show sequence
    for i in range(len(subactions)-1):
        plt.annotate('', xy=(i+1, 1), xytext=(i+1, 1),
                    arrowprops=dict(arrowstyle='->'))
    
    plt.title(f"Temporal sequence of subactions for '{action_name}'", fontsize=16)
    plt.ylim(0.5, 1.5)
    plt.xlim(-0.1, len(subactions) + 0.1)
    plt.axis('off')
    
    return plt.gcf(), None

# Function to generate verb and object vocabularies for ASKG
def generate_vocabulary_for_askg(action_labels):
    """Generate verb and object vocabularies from a given set of action labels"""
    # Extract verbs and objects from action labels
    verbs = []
    objects = []
    
    for label in action_labels:
        # Split the label into words
        words = label.split()
        
        # Simplified approach: first word is usually a verb, rest are objects
        if len(words) > 0:
            verbs.append(words[0])
            if len(words) > 1:
                objects.extend(words[1:])
    
    # Count occurrences
    verb_counter = Counter(verbs)
    object_counter = Counter(objects)
    
    return list(verb_counter.keys()), list(object_counter.keys())

# Main function to demonstrate all functionalities
def main():
    # Load ASKG data
    askg_data = load_askg_data('classes_ASKG_ntu_checked.yml')
    
    # 1. Generate t-SNE visualization
    tsne_skeleton = generate_tsne_visualization(askg_data, 'skeleton')
    tsne_semantic = generate_tsne_visualization(askg_data, 'semantic')
    
    # 2. Create object-subaction co-occurrence matrix
    matrix, subacts, objs = create_cooccurrence_matrix(askg_data, top_k_objects=10, top_k_subactions=10)
    cooccurrence_plot = plot_cooccurrence_matrix(matrix, subacts, objs)
    
    # 3. Create body part-subaction co-occurrence matrix
    bp_matrix, bp_subacts, body_parts = create_body_part_subaction_matrix(askg_data, top_k_subactions=20)
    bp_matrix_plot = plot_body_part_subaction_matrix(bp_matrix, bp_subacts, body_parts)
    
    # 4. Analyze and visualize subaction temporal relationships
    subaction_graph, all_subactions = analyze_subaction_temporal_relations(askg_data)
    temporal_network = plot_subaction_temporal_network(subaction_graph)
    
    # 5. Generate vocabularies
    object_vocab, subaction_vocab, body_part_vocab = create_vocabulary_from_existing_askg(askg_data)
    
    # Print results
    print(f"Number of actions in ASKG: {len(askg_data)}")
    print(f"Number of unique objects: {len(object_vocab)}")
    print(f"Number of unique subactions: {len(subaction_vocab)}")
    print(f"Number of unique body parts: {len(body_part_vocab)}")
    
    # Display visualizations
    tsne_skeleton.savefig('tsne_skeleton.png')
    tsne_semantic.savefig('tsne_semantic.png')
    cooccurrence_plot.savefig('cooccurrence_matrix.png')
    bp_matrix_plot.savefig('body_part_subaction_matrix.png')
    temporal_network.savefig('temporal_network.png')
    
    # Example subaction pattern visualization
    action_name = list(askg_data.keys())[0]  # First action
    pattern_plot, error_msg = visualize_subaction_patterns(askg_data, action_name)
    if pattern_plot:
        pattern_plot.savefig('subaction_pattern.png')
    else:
        print(f"Error: {error_msg}")

if __name__ == "__main__":
    main()