import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import random

def load_askg_data(file_path):
    """Load ASKG data from YAML file"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def extract_features(askg_data):
    """Extract features from ASKG data for t-SNE visualization"""
    # Extract objects and subactions
    objects = {}
    subactions = {}
    
    # First pass: collect all unique objects and subactions
    all_objects = set()
    all_subactions = set()
    
    for action_label, action_info in askg_data.items():
        # Extract objects
        if 'obj_li' in action_info and action_info['obj_li']:
            for obj in action_info['obj_li']:
                if isinstance(obj, dict):
                    # Handle case where obj is a dictionary with 'object' key
                    if 'object' in obj:
                        all_objects.add(obj['object'])
                else:
                    all_objects.add(obj)
        
        # Extract subactions
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                all_subactions.add(subact)
    
    # Create vocabulary mappings
    obj_to_idx = {obj: i for i, obj in enumerate(sorted(all_objects))}
    subact_to_idx = {subact: i for i, subact in enumerate(sorted(all_subactions))}
    
    # Second pass: create feature vectors
    action_features = {}
    
    for action_label, action_info in askg_data.items():
        # Initialize feature vectors
        obj_features = np.zeros(len(obj_to_idx))
        subact_features = np.zeros(len(subact_to_idx))
        
        # Fill object features
        if 'obj_li' in action_info and action_info['obj_li']:
            for obj in action_info['obj_li']:
                obj_name = obj
                if isinstance(obj, dict) and 'object' in obj:
                    obj_name = obj['object']
                
                if obj_name in obj_to_idx:
                    obj_features[obj_to_idx[obj_name]] = 1
        
        # Fill subaction features
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                if subact in subact_to_idx:
                    subact_features[subact_to_idx[subact]] = 1
        
        # Combine features
        combined_features = np.concatenate([obj_features, subact_features])
        action_features[action_label] = combined_features
    
    return action_features, obj_to_idx, subact_to_idx

def apply_tsne(features, perplexity=30, n_iter=1000):
    """Apply t-SNE dimensionality reduction"""
    # Prepare feature matrix
    feature_matrix = np.vstack(list(features.values()))
    
    # Check if we have enough samples for the requested perplexity
    n_samples = feature_matrix.shape[0]
    if n_samples <= perplexity:
        perplexity = max(5, n_samples - 1)
        print(f"Warning: Perplexity adjusted to {perplexity} due to small sample size")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings = tsne.fit_transform(feature_matrix)
    
    # Create a dictionary of label -> embedding
    embedding_dict = {label: embeddings[i] for i, label in enumerate(features.keys())}
    
    return embedding_dict

def split_seen_unseen(action_features, seed=42, seen_ratio=0.5):
    """Split actions into seen and unseen sets"""
    random.seed(seed)
    
    action_labels = list(action_features.keys())
    n_seen = int(len(action_labels) * seen_ratio)
    
    # Randomly select seen actions
    seen_labels = random.sample(action_labels, n_seen)
    unseen_labels = [label for label in action_labels if label not in seen_labels]
    
    # Create feature subsets
    seen_features = {label: action_features[label] for label in seen_labels}
    unseen_features = {label: action_features[label] for label in unseen_labels}
    
    return seen_features, unseen_features, seen_labels, unseen_labels

def visualize_tsne_result(embedding_dict, title="t-SNE Visualization", output_file=None):
    """Visualize t-SNE embeddings"""
    # Prepare data for plotting
    labels = list(embedding_dict.keys())
    embeddings = np.vstack(list(embedding_dict.values()))
    
    # Create a colormap
    n_classes = len(labels)
    cmap = plt.cm.get_cmap('tab20', n_classes)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot points
    for i, label in enumerate(labels):
        plt.scatter(embedding_dict[label][0], embedding_dict[label][1], 
                   color=cmap(i % 20), s=100, alpha=0.7)
    
    # Add title
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return plt.gcf()

def create_combined_visualization(seen_wo_prompt, seen_w_prompt, unseen_wo_prompt, unseen_w_prompt,
                                 seen_semantic_wo_prompt, seen_semantic_w_prompt, 
                                 unseen_semantic_wo_prompt, unseen_semantic_w_prompt,
                                 output_dir):
    """Create a combined visualization similar to Figure 4 in the reference image"""
    # Create figure with 8 subplots (2 rows, 4 columns)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    
    # First row: Skeleton space
    # (a) Known categories w/o prompt
    visualize_embeddings_in_subplot(axs[0, 0], seen_wo_prompt, 
                                   "(a) Known categories w/o prompt")
    
    # (b) Known categories w/ prompt
    visualize_embeddings_in_subplot(axs[0, 1], seen_w_prompt, 
                                   "(b) Known categories w/ prompt")
    
    # (c) Unknown categories w/o prompt
    visualize_embeddings_in_subplot(axs[0, 2], unseen_wo_prompt, 
                                   "(c) Unknown categories w/o prompt")
    
    # (d) Unknown categories w/ prompt
    visualize_embeddings_in_subplot(axs[0, 3], unseen_w_prompt, 
                                   "(d) Unknown categories w/ prompt")
    
    # Second row: Semantic space
    # (e) Known categories w/o prompt
    visualize_embeddings_in_subplot(axs[1, 0], seen_semantic_wo_prompt, 
                                   "(e) Known categories w/o prompt")
    
    # (f) Known categories w/ prompt
    visualize_embeddings_in_subplot(axs[1, 1], seen_semantic_w_prompt, 
                                   "(f) Known categories w/ prompt")
    
    # (g) Unknown categories w/o prompt
    visualize_embeddings_in_subplot(axs[1, 2], unseen_semantic_wo_prompt, 
                                   "(g) Unknown categories w/o prompt")
    
    # (h) Unknown categories w/ prompt
    visualize_embeddings_in_subplot(axs[1, 3], unseen_semantic_w_prompt, 
                                   "(h) Unknown categories w/ prompt")
    
    # Add a title for the entire figure
    fig.suptitle("Figure 4. t-SNE visualizations of skeleton and semantic spaces for known and unknown categories", 
                fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save figure
    fig.savefig(os.path.join(output_dir, "fig4_combined.png"), dpi=300)
    
    return fig

def visualize_embeddings_in_subplot(ax, embedding_dict, title):
    """Visualize embeddings in a subplot"""
    # Prepare data for plotting
    labels = list(embedding_dict.keys())
    embeddings = np.vstack(list(embedding_dict.values()))
    
    # Create a colormap
    n_classes = len(labels)
    cmap = plt.cm.get_cmap('tab20', n_classes)
    
    # Plot points
    for i, label in enumerate(labels):
        ax.scatter(embedding_dict[label][0], embedding_dict[label][1], 
                  color=cmap(i % 20), s=50, alpha=0.7)
    
    # Add title
    ax.set_title(title)
    ax.axis('off')

def create_semantic_features(features):
    """Create semantic features using cosine similarity"""
    # Compute pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Extract feature matrix
    feature_matrix = np.vstack(list(features.values()))
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Create semantic features dictionary
    semantic_features = {}
    for i, label in enumerate(features.keys()):
        semantic_features[label] = similarity_matrix[i]
    
    return semantic_features

def simulate_prompt_effect(features, noise_level=0.2, seed=42):
    """Simulate the effect of adding prompts to features"""
    np.random.seed(seed)
    
    # Create prompted features
    prompted_features = {}
    for label, feature in features.items():
        # Add random noise to simulate prompt effect
        noise = np.random.normal(0, noise_level, feature.shape)
        prompted_feature = feature + noise
        prompted_features[label] = prompted_feature
    
    return prompted_features

def main(askg_file, output_dir):
    """Main function for t-SNE visualization"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ASKG data
    print(f"Loading ASKG data from {askg_file}...")
    askg_data = load_askg_data(askg_file)
    print(f"Loaded {len(askg_data)} actions")
    
    # Extract features
    print("Extracting features...")
    action_features, obj_to_idx, subact_to_idx = extract_features(askg_data)
    print(f"Extracted features with {len(obj_to_idx)} objects and {len(subact_to_idx)} subactions")
    
    # Split into seen and unseen sets
    print("Splitting into seen and unseen sets...")
    seen_features, unseen_features, seen_labels, unseen_labels = split_seen_unseen(action_features)
    print(f"Split into {len(seen_features)} seen and {len(unseen_features)} unseen actions")
    
    # Create semantic features
    print("Creating semantic features...")
    seen_semantic_features = create_semantic_features(seen_features)
    unseen_semantic_features = create_semantic_features(unseen_features)
    
    # Simulate prompt effect
    print("Simulating prompt effect...")
    seen_features_prompted = simulate_prompt_effect(seen_features)
    unseen_features_prompted = simulate_prompt_effect(unseen_features)
    seen_semantic_features_prompted = simulate_prompt_effect(seen_semantic_features)
    unseen_semantic_features_prompted = simulate_prompt_effect(unseen_semantic_features)
    
    # Apply t-SNE to skeleton space
    print("Applying t-SNE to skeleton space...")
    seen_embeddings = apply_tsne(seen_features)
    seen_embeddings_prompted = apply_tsne(seen_features_prompted)
    unseen_embeddings = apply_tsne(unseen_features)
    unseen_embeddings_prompted = apply_tsne(unseen_features_prompted)
    
    # Apply t-SNE to semantic space
    print("Applying t-SNE to semantic space...")
    seen_semantic_embeddings = apply_tsne(seen_semantic_features)
    seen_semantic_embeddings_prompted = apply_tsne(seen_semantic_features_prompted)
    unseen_semantic_embeddings = apply_tsne(unseen_semantic_features)
    unseen_semantic_embeddings_prompted = apply_tsne(unseen_semantic_features_prompted)
    
    # Create individual visualizations
    print("Creating individual visualizations...")
    
    # Skeleton space
    visualize_tsne_result(seen_embeddings, 
                        title="(a) Known categories w/o prompt",
                        output_file=os.path.join(output_dir, "fig4a_known_wo_prompt.png"))
    
    visualize_tsne_result(seen_embeddings_prompted, 
                        title="(b) Known categories w/ prompt",
                        output_file=os.path.join(output_dir, "fig4b_known_w_prompt.png"))
    
    visualize_tsne_result(unseen_embeddings, 
                        title="(c) Unknown categories w/o prompt",
                        output_file=os.path.join(output_dir, "fig4c_unknown_wo_prompt.png"))
    
    visualize_tsne_result(unseen_embeddings_prompted, 
                        title="(d) Unknown categories w/ prompt",
                        output_file=os.path.join(output_dir, "fig4d_unknown_w_prompt.png"))
    
    # Semantic space
    visualize_tsne_result(seen_semantic_embeddings, 
                        title="(e) Known categories w/o prompt",
                        output_file=os.path.join(output_dir, "fig4e_known_wo_prompt_semantic.png"))
    
    visualize_tsne_result(seen_semantic_embeddings_prompted, 
                        title="(f) Known categories w/ prompt",
                        output_file=os.path.join(output_dir, "fig4f_known_w_prompt_semantic.png"))
    
    visualize_tsne_result(unseen_semantic_embeddings, 
                        title="(g) Unknown categories w/o prompt",
                        output_file=os.path.join(output_dir, "fig4g_unknown_wo_prompt_semantic.png"))
    
    visualize_tsne_result(unseen_semantic_embeddings_prompted, 
                        title="(h) Unknown categories w/ prompt",
                        output_file=os.path.join(output_dir, "fig4h_unknown_w_prompt_semantic.png"))
    
    # Create combined visualization
    print("Creating combined visualization...")
    create_combined_visualization(
        seen_embeddings, seen_embeddings_prompted, 
        unseen_embeddings, unseen_embeddings_prompted,
        seen_semantic_embeddings, seen_semantic_embeddings_prompted,
        unseen_semantic_embeddings, unseen_semantic_embeddings_prompted,
        output_dir
    )
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='t-SNE visualization of ASKG data')
    parser.add_argument('--askg', type=str, default='classes_ASKG_ntu_checked.yml',
                       help='Path to ASKG data file')
    parser.add_argument('--output', type=str, default='tsne_output',
                       help='Output directory')
    args = parser.parse_args()
    
    main(args.askg, args.output)