#!/usr/bin/env python3
import os
import argparse
import subprocess
import yaml
import sys
from pathlib import Path
"""
def check_dependencies():
    '''Check if required dependencies are installed'''
    required_packages = [
        'numpy', 'matplotlib', 'pandas', 'seaborn', 'scikit-learn', 
        'networkx', 'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True
"""
def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(output_dir):
    """Set up output directories"""
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        'tsne_visualizations', 
        'cooccurrence_matrices',
        'temporal_relations',
        'vocab_clustering',
        'vocab_constrained'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return {subdir: os.path.join(output_dir, subdir) for subdir in subdirs}

def run_tsne_visualization(askg_file, output_dir):
    """Run t-SNE visualization"""
    print("\n=== Running t-SNE Visualization ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    from askg_t_sne_visualization import main as tsne_main
    
    # Run t-SNE visualization
    tsne_main(askg_file, output_dir)
    
    print(f"t-SNE visualizations saved to {output_dir}")

def run_cooccurrence_analysis(askg_file, output_dir):
    """Run co-occurrence matrix analysis"""
    print("\n=== Running Co-occurrence Matrix Analysis ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    from askg_analysis import create_cooccurrence_matrix, plot_cooccurrence_matrix, load_askg_data
    
    # Load ASKG data
    askg_data = load_askg_data(askg_file)
    
    # Create co-occurrence matrix
    matrix, subacts, objs = create_cooccurrence_matrix(askg_data, top_k_objects=10, top_k_subactions=10)
    
    # Plot co-occurrence matrix
    plot_fig = plot_cooccurrence_matrix(matrix, subacts, objs, 
                           title="Co-occurrence Matrix - All Classes")
    
    # Save the figure manually
    output_file = os.path.join(output_dir, "cooccurrence_all.png")
    plot_fig.savefig(output_file)
    
    print(f"Co-occurrence matrix saved to {output_dir}")

def run_temporal_relation_analysis(askg_file, output_dir):
    """Run temporal relation analysis"""
    print("\n=== Running Temporal Relation Analysis ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    from temporal_relationship_analyzer import main as temporal_main
    
    # Run temporal relation analysis
    temporal_main(askg_file, output_dir)
    
    print(f"Temporal relation analysis saved to {output_dir}")

def run_vocabulary_generation(askg_file, labels_file, output_dir):
    """Run vocabulary generation"""
    print("\n=== Running Vocabulary Generation ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    from vocabulary_generator import main as vocab_main
    
    # Run vocabulary generation
    vocab_main(askg_file, labels_file, output_dir)
    
    print(f"Vocabulary generation outputs saved to {output_dir}")

def run_vocabulary_constrained(askg_file, labels_file, output_dir):
    """Run vocabulary constrained template generation"""
    print("\n=== Running Vocabulary Constrained Template Generation ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    from vocabulary_generator import load_askg_data, load_class_labels, extract_vocab_from_askg
    from vocabulary_generator import generate_clustered_vocabularies, generate_template_files
    
    # Load data
    askg_data = load_askg_data(askg_file)
    class_labels = load_class_labels(labels_file)
    
    print(f"Loaded ASKG with {len(askg_data)} actions")
    print(f"Loaded {len(class_labels)} class labels")
    
    # Generate vocabularies by clustering similar terms
    print("Generating vocabularies by clustering similar terms...")
    obj_clusters, subact_clusters, obj_mapping, subact_mapping = generate_clustered_vocabularies(
        askg_data, similarity_threshold=0.7)
    
    print(f"Generated {len(obj_clusters)} object clusters")
    print(f"Generated {len(subact_clusters)} subaction clusters")
    
    # Generate consolidated vocabularies
    consolidated_obj_vocab = list(obj_clusters.keys())
    consolidated_subact_vocab = list(subact_clusters.keys())
    
    # Generate template files for LLM-based ASKG generation
    print("Generating template files for LLM-based ASKG generation...")
    generate_template_files(class_labels, consolidated_obj_vocab, consolidated_subact_vocab, output_dir)
    
    # Save vocabularies in txt files for reference
    with open(os.path.join(output_dir, 'object_vocabulary.txt'), 'w') as f:
        for obj in consolidated_obj_vocab:
            f.write(f"{obj}\n")
    
    with open(os.path.join(output_dir, 'subaction_vocabulary.txt'), 'w') as f:
        for subact in consolidated_subact_vocab:
            f.write(f"{subact}\n")
            
    # Create a sample constrained ASKG for demonstration
    sample_constrained_askg = {}
    
    # Take first 5 classes as samples
    for label in class_labels[:5]:
        sample_objs = consolidated_obj_vocab[:3]  # Use first 3 objects as example
        sample_subacts = consolidated_subact_vocab[:3]  # Use first 3 subactions as example
        
        sample_constrained_askg[label] = {
            "label": label,
            "obj_li": sample_objs,
            "act_obj_triples": [
                f"<{label}, involves, {obj}>" for obj in sample_objs
            ],
            "sub_act_li": sample_subacts,
            "act_rel_triples": [
                f"<{sample_subacts[0]}, precedes, {sample_subacts[1]}>",
                f"<{sample_subacts[1]}, precedes, {sample_subacts[2]}>"
            ]
        }
    
    # Save sample constrained ASKG
    with open(os.path.join(output_dir, 'sample_constrained_askg.yml'), 'w') as f:
        yaml.dump(sample_constrained_askg, f, default_flow_style=False)
    
    print(f"Vocabulary constrained templates and sample ASKG saved to {output_dir}")

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ASKG Analysis and Visualization')
    parser.add_argument('--askg', type=str, default='classes_ASKG_ntu_checked.yml',
                       help='Path to ASKG data file')
    parser.add_argument('--labels', type=str, default='classes_label_ntu.yml',
                       help='Path to class labels file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--task', type=str, choices=['all', 'tsne', 'cooccurrence', 'temporal', 'vocabulary', 'constrained'],
                       default='all', help='Specific task to run')
    args = parser.parse_args()
    
    # Check dependencies
    # if not check_dependencies():
    #     sys.exit(1)
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Set up directories
    dirs = setup_directories(args.output)
    
    # Run tasks based on command-line arguments or configuration
    if args.task in ['all', 'tsne']:
        run_tsne_visualization(args.askg, dirs['tsne_visualizations'])
    
    if args.task in ['all', 'cooccurrence']:
        run_cooccurrence_analysis(args.askg, dirs['cooccurrence_matrices'])
    
    if args.task in ['all', 'temporal']:
        run_temporal_relation_analysis(args.askg, dirs['temporal_relations'])
    
    if args.task in ['all', 'vocabulary']:
        run_vocabulary_generation(args.askg, args.labels, dirs['vocab_clustering'])
    
    if args.task in ['all', 'constrained']:
        run_vocabulary_constrained(args.askg, args.labels, dirs['vocab_constrained'])
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to {args.output}")

if __name__ == "__main__":
    main()