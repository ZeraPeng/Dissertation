import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import argparse
from pathlib import Path

# Body part definitions
def get_part_names(body_part=4):
    if body_part == 6:
        part_names = ["head", "hand", "arm", "hip", "leg", "foot"]
    elif body_part == 4:
        part_names = ["head", "upper limbs", "hip", "lower limbs"]
    elif body_part == 2:
        part_names = ["upper body", "lower body"]
    return part_names

# Body part keywords for identifying if objects are body parts
BODY_PART_KEYWORDS = {
    "head": ["head", "face", "ear", "nose", "mouth", "neck", "eyes", "hair", "glasses"],
    "upper limbs": ["hand", "arm", "finger", "palm", "wrist", "elbow", "shoulder", "thumb"],
    "hip": ["hip", "waist", "torso", "chest", "back", "stomach"],
    "lower limbs": ["leg", "foot", "knee", "ankle", "toe", "feet", "thigh"]
}

# Objects to exclude
EXCLUDED_OBJECTS = ["air", "sound", "environment", "space", "distance", "surface", "balance"]

# Load ASKG data
def load_askg_data(file_path):
    """Load ASKG data from YAML file"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

# Load class labels
def load_class_labels(file_path):
    """Load action class labels from YAML file"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and 'classes' in data:
        return data['classes']
    return data

# Extract objects and subactions from ASKG
def extract_vocab_from_askg(askg_data):
    """Extract objects, subactions, and body parts from ASKG data"""
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
        
        # Extract subactions
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                subactions.append(subact)
    
    # Count occurrences
    object_counter = Counter(objects)
    subaction_counter = Counter(subactions)
    body_part_counter = Counter(list(body_parts))
    
    return object_counter, subaction_counter, body_part_counter

# Approach 1: Generate vocabularies from existing ASKG by clustering similar terms
def generate_clustered_vocabularies(askg_data, similarity_threshold=0.7):
    """
    Generate object and subaction vocabularies by clustering similar terms
    """
    # FIX: Unpack all three returned values
    object_counter, subaction_counter, body_part_counter = extract_vocab_from_askg(askg_data)
    
    # Get body part vocabulary
    body_part_vocab = get_part_names(4)  # Use 4-part body division
    
    # Get unique objects and subactions
    unique_objects = list(object_counter.keys())
    unique_subactions = list(subaction_counter.keys())
    
    # Calculate text similarity matrices
    # For objects
    obj_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
    obj_X = obj_vectorizer.fit_transform(unique_objects)
    obj_similarity = cosine_similarity(obj_X)
    
    # For subactions
    subact_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
    subact_X = subact_vectorizer.fit_transform(unique_subactions)
    subact_similarity = cosine_similarity(subact_X)
    
    # Cluster objects
    obj_clusters = defaultdict(list)
    obj_assigned = set()
    
    for i, obj1 in enumerate(unique_objects):
        if obj1 in obj_assigned:
            continue
        
        cluster = [obj1]
        obj_assigned.add(obj1)
        
        for j, obj2 in enumerate(unique_objects):
            if i != j and obj2 not in obj_assigned and obj_similarity[i, j] >= similarity_threshold:
                cluster.append(obj2)
                obj_assigned.add(obj2)
        
        # Use the most frequent object as the canonical form
        canonical_obj = max(cluster, key=lambda x: object_counter[x])
        obj_clusters[canonical_obj].extend(cluster)
    
    # Cluster subactions
    subact_clusters = defaultdict(list)
    subact_assigned = set()
    
    for i, subact1 in enumerate(unique_subactions):
        if subact1 in subact_assigned:
            continue
        
        cluster = [subact1]
        subact_assigned.add(subact1)
        
        for j, subact2 in enumerate(unique_subactions):
            if i != j and subact2 not in subact_assigned and subact_similarity[i, j] >= similarity_threshold:
                cluster.append(subact2)
                subact_assigned.add(subact2)
        
        # Use the most frequent subaction as the canonical form
        canonical_subact = max(cluster, key=lambda x: subaction_counter[x])
        subact_clusters[canonical_subact].extend(cluster)
    
    # Create mapping dictionaries
    obj_mapping = {}
    for canonical, cluster in obj_clusters.items():
        for obj in cluster:
            obj_mapping[obj] = canonical
    
    subact_mapping = {}
    for canonical, cluster in subact_clusters.items():
        for subact in cluster:
            subact_mapping[subact] = canonical
    
    # Handle unassigned terms
    for obj in unique_objects:
        if obj not in obj_mapping:
            obj_mapping[obj] = obj
            obj_clusters[obj] = [obj]
    
    for subact in unique_subactions:
        if subact not in subact_mapping:
            subact_mapping[subact] = subact
            subact_clusters[subact] = [subact]
    
    # Return clusters, body part vocab, and mappings
    return obj_clusters, subact_clusters, body_part_vocab, obj_mapping, subact_mapping

# Approach 2: Generate vocabularies directly from class labels
def generate_vocab_from_labels(class_labels, n_obj_vocab=50, n_subact_vocab=50):
    """
    Generate object and subaction vocabularies directly from class labels
    """
    # Extract verbs and nouns from labels
    verbs = []
    nouns = []
    
    for label in class_labels:
        words = label.split()
        if words:
            # Assume first word is a verb
            verbs.append(words[0])
            # Assume rest are nouns/objects
            if len(words) > 1:
                nouns.extend(words[1:])
    
    # Count occurrences
    verb_counter = Counter(verbs)
    noun_counter = Counter(nouns)
    
    # Select top N for each vocabulary
    obj_vocab = [noun for noun, _ in noun_counter.most_common(n_obj_vocab)]
    subact_vocab = [verb for verb, _ in verb_counter.most_common(n_subact_vocab)]
    
    return obj_vocab, subact_vocab

# Generate prompts for vocabulary-constrained ASKG
def generate_prompt_for_constrained_askg(action_label, obj_vocab, subact_vocab, body_part_vocab=None):
    """Generate prompt for creating vocabulary-constrained ASKG for a given action"""
    obj_list = "\n".join([f"- {obj}" for obj in obj_vocab])
    subact_list = "\n".join([f"- {subact}" for subact in subact_vocab])
    
    # Include body parts if provided
    body_part_section = ""
    if body_part_vocab:
        body_part_list = "\n".join([f"- {part}" for part in body_part_vocab])
        body_part_section = f"""
Body part vocabulary:
{body_part_list}
"""
        prompt_body_parts = """2. 1-2 most important body parts in this action (select from body part vocabulary)
"""
        yaml_body_parts = """  body_part_li:
  - [body_part1]
  - [body_part2]
  act_body_part_triples:
  - <{action_label}, [relation], [body_part1]>
  - <{action_label}, [relation], [body_part2]>
"""
    else:
        prompt_body_parts = ""
        yaml_body_parts = ""
    
    prompt = f"""Generate a knowledge graph for action "{action_label}" using ONLY terms from the provided object and subaction vocabularies.

Object vocabulary:
{obj_list}
{body_part_section}
Subaction vocabulary:
{subact_list}

Please generate:
1. 2-3 relevant objects for this action (select from object vocabulary)
{prompt_body_parts}3. 2-3 relevant subactions that make up this action (select from subaction vocabulary)
4. Action-object relation triples describing how the action relates to these objects
5. Subaction-subaction relation triples describing temporal relationships between subactions

Output format (YAML):
{action_label}:
  label: {action_label}
  obj_li:
  - [object1]
  - [object2]
{yaml_body_parts}  sub_act_li:
  - [subaction1]
  - [subaction2]
  sub_act_rel_triples:
  - <[subaction1], precedes, [subaction2]>
"""
    return prompt

# Save vocabularies to files
def save_vocabularies(obj_vocab, subact_vocab, output_dir):
    """Save generated vocabularies to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'object_vocabulary.txt'), 'w') as f:
        for obj in obj_vocab:
            f.write(f"{obj}\n")
    
    with open(os.path.join(output_dir, 'subaction_vocabulary.txt'), 'w') as f:
        for subact in subact_vocab:
            f.write(f"{subact}\n")
    
    print(f"Vocabularies saved to {output_dir}")

# Visualize vocabulary clusters
def visualize_vocab_clusters(obj_clusters, subact_clusters, output_dir):
    """Visualize vocabulary clusters"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    obj_cluster_data = []
    for canonical, cluster in obj_clusters.items():
        for obj in cluster:
            obj_cluster_data.append({
                'Canonical Form': canonical,
                'Term': obj,
                'Type': 'Object'
            })
    
    subact_cluster_data = []
    for canonical, cluster in subact_clusters.items():
        for subact in cluster:
            subact_cluster_data.append({
                'Canonical Form': canonical,
                'Term': subact,
                'Type': 'Subaction'
            })
    
    # Combine data
    all_data = pd.DataFrame(obj_cluster_data + subact_cluster_data)
    
    # Count cluster sizes
    obj_cluster_sizes = {canonical: len(cluster) for canonical, cluster in obj_clusters.items()}
    subact_cluster_sizes = {canonical: len(cluster) for canonical, cluster in subact_clusters.items()}
    
    # Visualize object clusters
    plt.figure(figsize=(14, 8))
    
    # Sort by cluster size
    sorted_obj_clusters = sorted(obj_cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    top_objs = [item[0] for item in sorted_obj_clusters[:15]]
    top_sizes = [item[1] for item in sorted_obj_clusters[:15]]
    
    plt.barh(top_objs, top_sizes, color='skyblue')
    plt.xlabel('Cluster Size')
    plt.ylabel('Canonical Object')
    plt.title('Top 15 Object Clusters by Size')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'object_clusters.png'))
    
    # Visualize subaction clusters
    plt.figure(figsize=(14, 8))
    
    # Sort by cluster size
    sorted_subact_clusters = sorted(subact_cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    top_subacts = [item[0] for item in sorted_subact_clusters[:15]]
    top_sizes = [item[1] for item in sorted_subact_clusters[:15]]
    
    plt.barh(top_subacts, top_sizes, color='lightgreen')
    plt.xlabel('Cluster Size')
    plt.ylabel('Canonical Subaction')
    plt.title('Top 15 Subaction Clusters by Size')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'subaction_clusters.png'))
    
    # Save clusters to CSV
    all_data.to_csv(os.path.join(output_dir, 'vocabulary_clusters.csv'), index=False)
    
    print(f"Cluster visualizations saved to {output_dir}")

# Apply vocabulary mappings to existing ASKG
def apply_vocab_mapping_to_askg(askg_data, obj_mapping, subact_mapping, output_dir):
    """Apply vocabulary mappings to existing ASKG and save the result"""
    # Create new ASKG with mapped vocabularies
    new_askg = {}
    
    for action_label, action_info in askg_data.items():
        new_action_info = {
            "label": action_info.get("label", action_label),
            "obj_li": [],
            "act_obj_triples": [],
            "sub_act_li": [],
            "act_rel_triples": []
        }
        
        # Map objects
        if 'obj_li' in action_info and action_info['obj_li']:
            mapped_objs = set()
            for obj in action_info['obj_li']:
                obj_name = obj
                if isinstance(obj, dict) and 'object' in obj:
                    obj_name = obj['object']
                
                if obj_name in obj_mapping:
                    mapped_obj = obj_mapping[obj_name]
                    mapped_objs.add(mapped_obj)
            
            new_action_info["obj_li"] = list(mapped_objs)
        
        # Map subactions
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            mapped_subacts = set()
            for subact in action_info['sub_act_li']:
                if subact in subact_mapping:
                    mapped_subact = subact_mapping[subact]
                    mapped_subacts.add(mapped_subact)
            
            new_action_info["sub_act_li"] = list(mapped_subacts)
        
        # Map action-object relation triples
        if 'act_obj_triples' in action_info and action_info['act_obj_triples']:
            for triple in action_info['act_obj_triples']:
                triple_str = str(triple)
                # Extract parts using regex
                match = re.search(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', triple_str)
                
                if match:
                    subj, rel, obj = match.groups()
                    subj = subj.strip()
                    rel = rel.strip()
                    obj = obj.strip()
                    
                    # Map object if it's in our mapping
                    if obj in obj_mapping:
                        mapped_obj = obj_mapping[obj]
                        new_triple = f"<{subj}, {rel}, {mapped_obj}>"
                        new_action_info["act_obj_triples"].append(new_triple)
        
        # Map action-action relation triples
        if 'act_rel_triples' in action_info and action_info['act_rel_triples']:
            for triple in action_info['act_rel_triples']:
                triple_str = str(triple)
                # Extract parts using regex
                match = re.search(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', triple_str)
                
                if match:
                    subj, rel, obj = match.groups()
                    subj = subj.strip()
                    rel = rel.strip()
                    obj = obj.strip()
                    
                    # Map subactions if they're in our mapping
                    mapped_subj = subact_mapping.get(subj, subj)
                    mapped_obj = subact_mapping.get(obj, obj)
                    
                    new_triple = f"<{mapped_subj}, {rel}, {mapped_obj}>"
                    new_action_info["act_rel_triples"].append(new_triple)
        
        # Add to new ASKG
        new_askg[action_label] = new_action_info
    
    # Save new ASKG
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'mapped_askg.yml'), 'w') as f:
        yaml.dump(new_askg, f, default_flow_style=False)
    
    print(f"Mapped ASKG saved to {os.path.join(output_dir, 'mapped_askg.yml')}")
    
    return new_askg

def get_safe_filename(label):
    """Convert a string to a safe filename by removing/replacing problematic characters"""
    # Replace spaces and other potentially problematic characters
    safe_name = re.sub(r'[/\\?%*:|"<>]', '_', label)
    # Replace spaces with underscores
    safe_name = safe_name.replace(' ', '_')
    return safe_name

# Generate template files
def generate_template_files(class_labels, obj_vocab, subact_vocab, body_part_vocab, output_dir):
    """Generate template files for LLM-based ASKG generation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate template for each class
    for i, label in enumerate(class_labels):
        prompt = generate_prompt_for_constrained_askg(label, obj_vocab, subact_vocab, body_part_vocab)
        
        # Use safe filename
        safe_name = get_safe_filename(label)
        with open(os.path.join(output_dir, f"{safe_name}_template.txt"), 'w') as f:
            f.write(prompt)
    
    # Generate batch template file
    batch_prompt = "Generate knowledge graphs for the following actions using ONLY terms from the provided object and subaction vocabularies.\n\n"
    
    # Add vocabularies
    obj_list = "\n".join([f"- {obj}" for obj in obj_vocab])
    subact_list = "\n".join([f"- {subact}" for subact in subact_vocab])
    body_part_list = "\n".join([f"- {part}" for part in body_part_vocab])
    
    batch_prompt += f"Object vocabulary:\n{obj_list}\n\n"
    batch_prompt += f"Body part vocabulary:\n{body_part_list}\n\n"
    batch_prompt += f"Subaction vocabulary:\n{subact_list}\n\n"
    
    # Add instructions
    batch_prompt += """For each action, please generate:
1. 2-3 relevant objects from the object vocabulary that are involved in the action
2. 1-2 most important body parts involved in the action (from body part vocabulary)
3. 2-3 relevant subactions from the subaction vocabulary that make up the action
4. Action-object relation triples that describe how the action relates to these objects
5. Action-body part relation triples that describe how the action relates to body parts
6. Subaction-subaction relation triples that describe the temporal relationships between subactions

Output format (YAML):"""
    
    # Add example
    example_label = class_labels[0] if class_labels else "action_label"
    batch_prompt += f"""

{example_label}:
  label: {example_label}
  obj_li:
  - [object1]
  - [object2]
  body_part_li:
  - [body_part1]
  - [body_part2]
  act_obj_triples:
  - <{example_label}, [relation], [object1]>
  - <{example_label}, [relation], [object2]>
  act_body_part_triples:
  - <{example_label}, [relation], [body_part1]>
  - <{example_label}, [relation], [body_part2]>
  sub_act_li:
  - [subaction1]
  - [subaction2]
  sub_act_rel_triples:
  - <[subaction1], precedes, [subaction2]>

Now generate the ASKG for the following actions:
"""
    
    # Add action labels
    for label in class_labels:
        batch_prompt += f"- {label}\n"
    
    with open(os.path.join(output_dir, "batch_template.txt"), 'w') as f:
        f.write(batch_prompt)
    
    print(f"Template files generated in {output_dir}")
    

# Compare original and mapped ASKGs
def compare_askg_stats(original_askg, mapped_askg):
    """Compare statistics between original and mapped ASKGs"""
    # Extract original vocabularies
    orig_obj_counter, orig_subact_counter, _ = extract_vocab_from_askg(original_askg)
    
    # Extract mapped vocabularies
    mapped_obj_counter, mapped_subact_counter, _ = extract_vocab_from_askg(mapped_askg)
    
    # Prepare comparison data
    comparison_data = [
        ["Original Objects", len(orig_obj_counter)],
        ["Mapped Objects", len(mapped_obj_counter)],
        ["Original Subactions", len(orig_subact_counter)],
        ["Mapped Subactions", len(mapped_subact_counter)]
    ]
    
    # Calculate reduction percentages
    obj_reduction = (1 - len(mapped_obj_counter) / len(orig_obj_counter)) * 100
    subact_reduction = (1 - len(mapped_subact_counter) / len(orig_subact_counter)) * 100
    
    comparison_data.append(["Object Vocabulary Reduction", f"{obj_reduction:.2f}%"])
    comparison_data.append(["Subaction Vocabulary Reduction", f"{subact_reduction:.2f}%"])
    
    # Convert to DataFrame for pretty printing
    df = pd.DataFrame(comparison_data, columns=["Metric", "Value"])
    
    return df

# Create method to generate a compact representation of ASKG for generation system
def create_compact_askg_representation(class_labels, obj_vocab, subact_vocab, body_part_vocab=None):
    """Create a compact representation of ASKG for generation system"""
    # Create a dictionary to store the representation
    compact_repr = {
        "class_labels": class_labels,
        "object_vocabulary": obj_vocab,
        "subaction_vocabulary": subact_vocab
    }
    
    # Add body part vocabulary if provided
    if body_part_vocab:
        compact_repr["body_part_vocabulary"] = body_part_vocab
    
    return compact_repr

# Main function
def main(askg_file=None, labels_file=None, output_dir=None):
    """Main function"""
    # Parse command-line arguments if not provided
    if askg_file is None or labels_file is None or output_dir is None:
        parser = argparse.ArgumentParser(description='Generate vocabularies for ASKG')
        parser.add_argument('--askg', type=str, default='classes_ASKG_ntu_checked.yml',
                           help='Path to ASKG data file')
        parser.add_argument('--labels', type=str, default='classes_label_ntu.yml',
                           help='Path to class labels file')
        parser.add_argument('--output', type=str, default='output',
                           help='Output directory')
        parser.add_argument('--similarity', type=float, default=0.7,
                           help='Similarity threshold for clustering')
        args = parser.parse_args()
        
        # Use command-line arguments if not provided
        if askg_file is None:
            askg_file = args.askg
        if labels_file is None:
            labels_file = args.labels
        if output_dir is None:
            output_dir = args.output
        
        similarity_threshold = args.similarity
    else:
        # Use default similarity threshold
        similarity_threshold = 0.7
    
    # Load data
    askg_data = load_askg_data(askg_file)
    class_labels = load_class_labels(labels_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loaded ASKG with {len(askg_data)} actions")
    print(f"Loaded {len(class_labels)} class labels")
    
    # Method 1: Generate vocabularies by clustering similar terms
    print("\nGenerating vocabularies by clustering similar terms...")
    obj_clusters, subact_clusters, body_part_vocab, obj_mapping, subact_mapping = generate_clustered_vocabularies(
        askg_data, similarity_threshold=similarity_threshold)
    
    print(f"Generated {len(obj_clusters)} object clusters")
    print(f"Generated {len(subact_clusters)} subaction clusters")
    print(f"Using {len(body_part_vocab)} body parts: {', '.join(body_part_vocab)}")
    
    # Visualize clusters
    visualize_vocab_clusters(obj_clusters, subact_clusters, os.path.join(output_dir, 'clusters'))
    
    # Generate consolidated vocabularies
    consolidated_obj_vocab = list(obj_clusters.keys())
    consolidated_subact_vocab = list(subact_clusters.keys())
    
    # Save vocabularies
    save_vocabularies(consolidated_obj_vocab, consolidated_subact_vocab, 
                     os.path.join(output_dir, 'consolidated_vocab'))
    
    # Save body part vocabulary
    with open(os.path.join(output_dir, 'consolidated_vocab', 'body_part_vocabulary.txt'), 'w') as f:
        for part in body_part_vocab:
            f.write(f"{part}\n")
    
    # Apply mapping to existing ASKG
    mapped_askg = apply_vocab_mapping_to_askg(askg_data, obj_mapping, subact_mapping, 
                                            os.path.join(output_dir, 'mapped_askg'))
    
    # Method 2: Generate vocabularies directly from class labels
    print("\nGenerating vocabularies from class labels...")
    direct_obj_vocab, direct_subact_vocab = generate_vocab_from_labels(class_labels)
    
    print(f"Generated {len(direct_obj_vocab)} objects and {len(direct_subact_vocab)} subactions from class labels")
    
    # Save direct vocabularies
    save_vocabularies(direct_obj_vocab, direct_subact_vocab, 
                     os.path.join(output_dir, 'direct_vocab'))
    
    # Create consolidated representation
    compact_repr = create_compact_askg_representation(class_labels, consolidated_obj_vocab, 
                                                    consolidated_subact_vocab, body_part_vocab)
    
    # Save compact representation
    with open(os.path.join(output_dir, 'compact_askg_repr.json'), 'w') as f:
        json.dump(compact_repr, f, indent=2)
    
    # Generate template files for LLM-based ASKG generation
    print("\nGenerating template files for LLM-based ASKG generation...")
    templates_dir = os.path.join(output_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    generate_template_files(class_labels, consolidated_obj_vocab, consolidated_subact_vocab, 
                          body_part_vocab, templates_dir)
    
    # Compare statistics
    print("\nComparing original and mapped ASKGs...")
    comparison_df = compare_askg_stats(askg_data, mapped_askg)
    print(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(output_dir, 'askg_comparison.csv'), index=False)
    
    print(f"\nAll outputs saved to {output_dir}")
    print("\nTo use the vocabulary-constrained ASKG system:")
    print(f"1. Check the vocabularies in the '{os.path.join(output_dir, 'consolidated_vocab')}' directory")
    print(f"2. Review the template files in the '{os.path.join(output_dir, 'templates')}' directory")
    print(f"3. Use the templates with an LLM to generate vocabulary-constrained ASKGs")
    print(f"4. Compare the results with the mapped ASKG in the '{os.path.join(output_dir, 'mapped_askg')}' directory")

if __name__ == "__main__":
    main()