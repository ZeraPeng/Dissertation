import os
import argparse
import subprocess
import yaml
import sys
import json
import time
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

# Add body part definitions
def get_part_names(body_part=4):
    if body_part == 6:
        part_names = ["head", "hand", "arm", "hip", "leg", "foot"]
    elif body_part == 4:
        part_names = ["head", "upper limbs", "hip", "lower limbs"]
    elif body_part == 2:
        part_names = ["upper body", "lower body"]
    return part_names

# Body part keywords - for identifying if objects are body parts
BODY_PART_KEYWORDS = {
    "head": ["head", "face", "ear", "nose", "mouth", "neck", "eyes", "hair", "glasses"],
    "upper limbs": ["hand", "arm", "finger", "palm", "wrist", "elbow", "shoulder", "thumb"],
    "hip": ["hip", "waist", "torso", "chest", "back", "stomach"],
    "lower limbs": ["leg", "foot", "knee", "ankle", "toe", "feet", "thigh"]
}

# Objects to exclude
EXCLUDED_OBJECTS = ["air", "sound", "environment", "space", "distance", "surface", "balance"]

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
        'body_part_matrices',  # Added: body part matrices
        'temporal_relations',
        'vocab_clustering',
        'enhanced_askg',  # Added: enhanced ASKG
        'vocab_constrained',
        'vocab_constrained/templates',
        'vocab_constrained/visualizations',
        'vocab_constrained/sample_outputs'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return {subdir.split('/')[-1] if '/' in subdir else subdir: 
            os.path.join(output_dir, subdir) for subdir in subdirs}

def run_body_part_analysis(askg_file, output_dir):
    """Run body part-subaction matrix analysis"""
    print("\n=== Running Body Part-Subaction Matrix Analysis ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    from askg_analysis import load_askg_data
    
    # Check if create_body_part_subaction_matrix function exists
    try:
        from askg_analysis import create_body_part_subaction_matrix, plot_body_part_subaction_matrix
        
        # Load ASKG data
        askg_data = load_askg_data(askg_file)
        
        # Use imported function to create body part-subaction co-occurrence matrix
        matrix, subacts, body_parts = create_body_part_subaction_matrix(askg_data, top_k_subactions=20)
        
        # Plot body part-subaction co-occurrence matrix
        plot_fig = plot_body_part_subaction_matrix(matrix, subacts, body_parts, 
                               title="Body Part-Subaction Co-occurrence Matrix")
    
    except (ImportError, AttributeError, TypeError):
        print("Warning: Unable to use imported body part analysis functions. Using inline implementation.")
        
        # Load ASKG data
        askg_data = load_askg_data(askg_file)
        
        # Inline implementation of body part-subaction co-occurrence matrix
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get body part vocabulary
        body_parts = get_part_names(4)  # Use 4-part body division
        
        # Get all subactions
        all_subactions = set()
        for action_label, action_info in askg_data.items():
            if 'sub_act_li' in action_info and action_info['sub_act_li']:
                for subact in action_info['sub_act_li']:
                    all_subactions.add(subact)
        
        # Sort alphabetically
        all_subactions = sorted(list(all_subactions))
        
        # Limit number of subactions
        top_k_subactions = min(20, len(all_subactions))
        top_subactions = all_subactions[:top_k_subactions]
        
        # Initialize co-occurrence matrix
        matrix = np.zeros((len(top_subactions), len(body_parts)))
        
        # Fill matrix
        for action_label, action_info in askg_data.items():
            # Identify body parts for this action
            action_body_parts = []
            if 'obj_li' in action_info and action_info['obj_li']:
                for obj in action_info['obj_li']:
                    obj_name = obj
                    if isinstance(obj, dict) and 'object' in obj:
                        obj_name = obj['object']
                    
                    # Check if it's a body part
                    for part, keywords in BODY_PART_KEYWORDS.items():
                        if any(keyword in obj_name.lower() for keyword in keywords):
                            if part in body_parts and part not in action_body_parts:
                                action_body_parts.append(part)
                            break
            
            # Get subactions for this action
            action_subactions = []
            if 'sub_act_li' in action_info and action_info['sub_act_li']:
                for subact in action_info['sub_act_li']:
                    if subact in top_subactions:
                        action_subactions.append(subact)
            
            # Update co-occurrence matrix
            for subact in action_subactions:
                for body_part in action_body_parts:
                    subact_idx = top_subactions.index(subact)
                    body_part_idx = body_parts.index(body_part)
                    matrix[subact_idx, body_part_idx] += 1
        
        # Inline implementation of plot function
        def plot_body_part_subaction_matrix(matrix, row_labels, col_labels, title="Body Part-Subaction Co-occurrence Matrix"):
            plt.figure(figsize=(12, 10))
            
            # Create custom colormap
            cmap = sns.color_palette("YlGnBu", as_cmap=True)
            
            # Convert matrix to integer type
            matrix = np.array(matrix).astype(int)
            
            # Plot heatmap
            sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap,
                        xticklabels=col_labels, yticklabels=row_labels)
            
            plt.title(title, fontsize=16)
            plt.xlabel('Body Parts', fontsize=14)
            plt.ylabel('Subactions', fontsize=14)
            plt.tight_layout()
            
            return plt.gcf()
        
        # Plot co-occurrence matrix
        plot_fig = plot_body_part_subaction_matrix(matrix, top_subactions, body_parts)
        
        # Use inline implementation results
        subacts = top_subactions
    
    # Save figure
    output_file = os.path.join(output_dir, "body_part_subaction_matrix.png")
    plot_fig.savefig(output_file)
    
    print(f"Body part-subaction matrix saved to {output_dir}")
    
    return matrix, subacts, body_parts

def run_enhanced_askg_generation(askg_file, output_dir):
    """Run enhanced ASKG generation"""
    print("\n=== Generating Enhanced ASKG (with body parts) ===")
    
    # Import necessary modules
    import sys
    sys.path.append('.')
    
    try:
        # Try to import enhanced ASKG generator
        from create_enhanced_askg import main as enhanced_askg_main
        
        # Run enhanced ASKG generation
        enhanced_askg_main(askg_file, output_dir)
        
        print(f"Enhanced ASKG saved to {output_dir}")
        
        # Return path to generated enhanced ASKG file
        return os.path.join(output_dir, "enhanced_askg.yml")
    
    except ImportError:
        print("Warning: Unable to import create_enhanced_askg module. Using inline implementation.")
        
        # Inline implementation of enhanced ASKG generation
        import yaml
        
        # Load ASKG data
        with open(askg_file, 'r') as f:
            askg_data = yaml.safe_load(f)
        
        # Create enhanced ASKG
        enhanced_askg = {}
        
        # Iterate through original ASKG data
        for action_label, action_info in askg_data.items():
            # Create enhanced version of action info
            enhanced_action = {
                "label": action_info.get("label", action_label),
                "obj_li": [],
                "body_part_li": [],
                "act_obj_triples": [],
                "act_body_part_triples": [],
                "sub_act_li": action_info.get("sub_act_li", []),
                "sub_act_rel_triples": action_info.get("sub_act_rel_triples", [])
            }
            
            # Process objects and body parts
            if 'obj_li' in action_info and action_info['obj_li']:
                obj_list = []
                body_parts = []
                
                for obj in action_info['obj_li']:
                    obj_name = obj
                    if isinstance(obj, dict) and 'object' in obj:
                        obj_name = obj['object']
                    
                    # Check if it's a body part
                    is_body_part = False
                    for part, keywords in BODY_PART_KEYWORDS.items():
                        if any(keyword in obj_name.lower() for keyword in keywords):
                            body_parts.append(part)
                            is_body_part = True
                            break
                    
                    # If not a body part and not in exclusion list
                    if not is_body_part and obj_name.lower() not in EXCLUDED_OBJECTS:
                        obj_list.append(obj_name)
                
                enhanced_action["obj_li"] = obj_list
                enhanced_action["body_part_li"] = list(set(body_parts))  # Deduplicate
            
            # Process triples
            if 'act_obj_triples' in action_info and action_info['act_obj_triples']:
                # Simplified version, in practice should have similar logic to main function
                enhanced_action["act_obj_triples"] = action_info['act_obj_triples']
            
            # Add to enhanced ASKG
            enhanced_askg[action_label] = enhanced_action
        
        # Save enhanced ASKG
        enhanced_askg_file = os.path.join(output_dir, "enhanced_askg.yml")
        with open(enhanced_askg_file, 'w') as f:
            yaml.dump(enhanced_askg, f, default_flow_style=False)
        
        return enhanced_askg_file

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

def create_realistic_example(action_label, objs, subacts, relevant_obj_maps=None, relevant_subact_maps=None):
    """Create a more realistic example ASKG for a specific action"""
    # Select relevant objects and subactions for this specific action
    if relevant_obj_maps and action_label in relevant_obj_maps:
        selected_objs = relevant_obj_maps[action_label][:min(3, len(relevant_obj_maps[action_label]))]
    else:
        # Default to first 3 objects if no mapping is available
        selected_objs = objs[:min(3, len(objs))]
    
    if relevant_subact_maps and action_label in relevant_subact_maps:
        selected_subacts = relevant_subact_maps[action_label][:min(3, len(relevant_subact_maps[action_label]))]
    else:
        # Default to first 3 subactions if no mapping is available
        selected_subacts = subacts[:min(3, len(subacts))]
    
    # Create action-object triples with more meaningful relations based on action type
    act_obj_triples = []
    for obj in selected_objs:
        relation = "involves"
        
        # Create more specific relations based on action and object patterns
        if "take" in action_label or "put" in action_label:
            relation = "manipulates"
        elif "eat" in action_label or "drink" in action_label:
            relation = "consumes"
        elif "throw" in action_label:
            relation = "propels"
        elif "wear" in action_label or "jacket" in action_label or "hat" in action_label:
            relation = "wears"
        
        act_obj_triples.append(f"<{action_label}, {relation}, {obj}>")
    
    # Create subaction relation triples with more meaningful temporal relations
    act_rel_triples = []
    if len(selected_subacts) >= 2:
        # Create a linear sequence of subactions
        for i in range(len(selected_subacts) - 1):
            act_rel_triples.append(f"<{selected_subacts[i]}, precedes, {selected_subacts[i+1]}>")
    
    return {
        "label": action_label,
        "obj_li": selected_objs,
        "act_obj_triples": act_obj_triples,
        "sub_act_li": selected_subacts,
        "act_rel_triples": act_rel_triples
    }

def extract_relevant_terms(askg_data, obj_mapping, subact_mapping):
    """Extract relevant objects and subactions for each action"""
    relevant_objs = {}
    relevant_subacts = {}
    
    for action_label, action_info in askg_data.items():
        relevant_objs[action_label] = []
        relevant_subacts[action_label] = []
        
        # Extract objects
        if 'obj_li' in action_info and action_info['obj_li']:
            for obj in action_info['obj_li']:
                obj_name = obj
                if isinstance(obj, dict) and 'object' in obj:
                    obj_name = obj['object']
                
                if obj_name in obj_mapping:
                    mapped_obj = obj_mapping[obj_name]
                    if mapped_obj not in relevant_objs[action_label]:
                        relevant_objs[action_label].append(mapped_obj)
        
        # Extract subactions
        if 'sub_act_li' in action_info and action_info['sub_act_li']:
            for subact in action_info['sub_act_li']:
                if subact in subact_mapping:
                    mapped_subact = subact_mapping[subact]
                    if mapped_subact not in relevant_subacts[action_label]:
                        relevant_subacts[action_label].append(mapped_subact)
    
    return relevant_objs, relevant_subacts

def generate_template_files_parallel(class_labels, obj_vocab, subact_vocab, output_dir, batch_size=10):
    """Generate template files for LLM-based ASKG generation using parallel processing"""
    from vocabulary_generator import generate_prompt_for_constrained_askg, get_safe_filename
    
    os.makedirs(output_dir, exist_ok=True)
    templates_dir = os.path.join(output_dir, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Function to generate and save a template for a specific class
    def process_class(label):
        prompt = generate_prompt_for_constrained_askg(label, obj_vocab, subact_vocab)
        filename = os.path.join(templates_dir, f"{get_safe_filename(label)}_template.txt")
        with open(filename, 'w') as f:
            f.write(prompt)
        return label
    
    # Generate templates in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
        futures = {executor.submit(process_class, label): label for label in class_labels}
        
        # Use tqdm for progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Generating templates"):
            results.append(future.result())
    
    # Generate batch template files
    num_batches = (len(class_labels) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(class_labels))
        batch_labels = class_labels[start_idx:end_idx]
        
        batch_prompt = "Generate knowledge graphs for the following actions using ONLY terms from the provided object and subaction vocabularies.\n\n"
        
        # Add vocabularies
        obj_list = "\n".join([f"- {obj}" for obj in obj_vocab])
        subact_list = "\n".join([f"- {subact}" for subact in subact_vocab])
        
        batch_prompt += f"Object Vocabulary:\n{obj_list}\n\n"
        batch_prompt += f"Subaction Vocabulary:\n{subact_list}\n\n"
        
        # Add instructions
        batch_prompt += """For each action, please generate:
1. 2-3 relevant objects from the object vocabulary that are involved in the action
2. Action-object relation triples that describe how the action relates to these objects
3. 2-3 relevant subactions from the subaction vocabulary that make up the action
4. Subaction-subaction relation triples that describe the temporal relationships between subactions

Output format (YAML):"""
        
        # Add example
        example_label = batch_labels[0]
        batch_prompt += f"""

{example_label}:
  label: {example_label}
  obj_li:
  - [object1]
  - [object2]
  act_obj_triples:
  - <{example_label}, [relation], [object1]>
  - <{example_label}, [relation], [object2]>
  sub_act_li:
  - [subaction1]
  - [subaction2]
  act_rel_triples:
  - <[subaction1], precedes, [subaction2]>

Now generate the ASKG for the following actions:
"""
        
        # Add action labels
        for label in batch_labels:
            batch_prompt += f"- {label}\n"
        
        batch_filename = os.path.join(output_dir, f"batch_{batch_idx+1}_template.txt")
        with open(batch_filename, 'w') as f:
            f.write(batch_prompt)
    
    return results

def run_vocabulary_constrained(askg_file, labels_file, output_dir, similarity_threshold=0.7):
    """Run vocabulary constrained template generation, enhanced functionality for body parts"""
    print("\n=== Running Vocabulary Constrained Template Generation ===")
    start_time = time.time()
    
    # Import necessary modules
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sys.path.append('.')
    from vocabulary_generator import (
        load_askg_data, load_class_labels, extract_vocab_from_askg,
        generate_clustered_vocabularies, generate_template_files,
        get_safe_filename
    )
    
    # Load data
    askg_data = load_askg_data(askg_file)
    class_labels = load_class_labels(labels_file)
    
    print(f"Loaded ASKG with {len(askg_data)} actions")
    print(f"Loaded {len(class_labels)} class labels")
    
    # Generate vocabularies by clustering similar terms
    print("Generating vocabularies by clustering similar terms...")
    
    # Check extract_vocab_from_askg's parameters and return values
    import inspect
    sig = inspect.signature(extract_vocab_from_askg)
    
    # Get body part vocabulary
    body_part_vocab = get_part_names(4)  # Use 4-part body division
    
    # If function supports extracting body parts (parameter count matches)
    if len(sig.parameters) == 1:
        try:
            # Try to call function, check if it returns 3 values (including body parts)
            result = extract_vocab_from_askg(askg_data)
            if len(result) == 3:
                obj_counter, subact_counter, body_part_counter = result
                has_body_part_support = True
            else:
                obj_counter, subact_counter = result
                has_body_part_support = False
        except Exception as e:
            print(f"Warning: Error extracting vocabulary: {e}")
            # Create default counters
            from collections import Counter
            obj_counter = Counter()
            subact_counter = Counter()
            has_body_part_support = False
    else:
        # Function signature doesn't match
        print("Warning: extract_vocab_from_askg function signature doesn't match current requirements")
        # Create default counters
        from collections import Counter
        obj_counter = Counter()
        subact_counter = Counter()
        has_body_part_support = False
    
    # Check generate_clustered_vocabularies's parameters and return values
    sig = inspect.signature(generate_clustered_vocabularies)
    
    # If function supports clustering (parameter count matches)
    if len(sig.parameters) >= 2:
        try:
            # Try to call function
            result = generate_clustered_vocabularies(askg_data, similarity_threshold=similarity_threshold)
            if len(result) >= 4:  # Enough return values
                if len(result) == 5:  # New version returns 5 values
                    obj_clusters, subact_clusters, bp_vocab, obj_mapping, subact_mapping = result
                else:  # Old version returns 4 values
                    obj_clusters, subact_clusters, obj_mapping, subact_mapping = result
                    bp_vocab = body_part_vocab  # Use default body part vocab
            else:
                print("Warning: generate_clustered_vocabularies doesn't return enough values")
                # Create default values
                obj_clusters = {}
                subact_clusters = {}
                obj_mapping = {}
                subact_mapping = {}
                bp_vocab = body_part_vocab
        except Exception as e:
            print(f"Warning: Error clustering vocabulary: {e}")
            # Create default values
            obj_clusters = {}
            subact_clusters = {}
            obj_mapping = {}
            subact_mapping = {}
            bp_vocab = body_part_vocab
    else:
        # Function signature doesn't match
        print("Warning: generate_clustered_vocabularies function signature doesn't match current requirements")
        # Create default values
        obj_clusters = {}
        subact_clusters = {}
        obj_mapping = {}
        subact_mapping = {}
        bp_vocab = body_part_vocab
    
    print(f"Generated {len(obj_clusters)} object clusters")
    print(f"Generated {len(subact_clusters)} subaction clusters")
    print(f"Using {len(bp_vocab)} body parts: {', '.join(bp_vocab)}")
    
    # Generate consolidated vocabularies
    consolidated_obj_vocab = list(obj_clusters.keys()) if obj_clusters else []
    consolidated_subact_vocab = list(subact_clusters.keys()) if subact_clusters else []
    
    # Extract relevant terms for each action
    try:
        # Try to call extract_relevant_terms function (if exists)
        relevant_objs, relevant_subacts = extract_relevant_terms(askg_data, obj_mapping, subact_mapping)
    except NameError:
        # If function doesn't exist, define a simplified version
        def extract_relevant_terms(askg_data, obj_mapping, subact_mapping):
            """Extract relevant objects and subactions for each action"""
            relevant_objs = {}
            relevant_subacts = {}
            
            for action_label, action_info in askg_data.items():
                relevant_objs[action_label] = []
                relevant_subacts[action_label] = []
                
                # Extract objects
                if 'obj_li' in action_info and action_info['obj_li']:
                    for obj in action_info['obj_li']:
                        obj_name = obj
                        if isinstance(obj, dict) and 'object' in obj:
                            obj_name = obj['object']
                        
                        # Use mapping (if exists)
                        if obj_name in obj_mapping:
                            mapped_obj = obj_mapping[obj_name]
                            if mapped_obj not in relevant_objs[action_label]:
                                relevant_objs[action_label].append(mapped_obj)
                
                # Extract subactions
                if 'sub_act_li' in action_info and action_info['sub_act_li']:
                    for subact in action_info['sub_act_li']:
                        if subact in subact_mapping:
                            mapped_subact = subact_mapping[subact]
                            if mapped_subact not in relevant_subacts[action_label]:
                                relevant_subacts[action_label].append(mapped_subact)
            
            return relevant_objs, relevant_subacts
        
        # Use simplified function
        relevant_objs, relevant_subacts = extract_relevant_terms(askg_data, obj_mapping, subact_mapping)
    
    # Define enhanced template generation function with body parts
    def generate_enhanced_template(action_label, obj_vocab, subact_vocab, body_part_vocab):
        """Generate enhanced template with body parts"""
        obj_list = "\n".join([f"- {obj}" for obj in obj_vocab])
        subact_list = "\n".join([f"- {subact}" for subact in subact_vocab])
        body_part_list = "\n".join([f"- {part}" for part in body_part_vocab])
        
        prompt = f"""Generate a knowledge graph for action "{action_label}" using ONLY terms from the provided object, body part, and subaction vocabularies.

Object vocabulary:
{obj_list}

Body part vocabulary:
{body_part_list}

Subaction vocabulary:
{subact_list}

Please generate:
1. 2-3 relevant objects for this action (select from object vocabulary)
2. 1-2 most important body parts in this action (select from body part vocabulary)
3. 2-3 relevant subactions that make up this action (select from subaction vocabulary)
4. Action-object relation triples describing how the action relates to these objects
5. Action-body part relation triples describing how the action relates to body parts
6. Subaction-subaction relation triples describing temporal relationships between subactions

Output format (YAML):
{action_label}:
  label: {action_label}
  obj_li:
  - [object1]
  - [object2]
  body_part_li:
  - [body_part1]
  - [body_part2]
  act_obj_triples:
  - <{action_label}, [relation], [object1]>
  - <{action_label}, [relation], [object2]>
  act_body_part_triples:
  - <{action_label}, [relation], [body_part1]>
  - <{action_label}, [relation], [body_part2]>
  sub_act_li:
  - [subaction1]
  - [subaction2]
  sub_act_rel_triples:
  - <[subaction1], precedes, [subaction2]>
"""
        return prompt
    
    # Define parallel template generation function
    def generate_template_files_parallel(class_labels, obj_vocab, subact_vocab, body_part_vocab, output_dir, batch_size=10):
        """Generate template files using parallel processing"""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        os.makedirs(output_dir, exist_ok=True)
        templates_dir = os.path.join(output_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        # Function to process a single class
        def process_class(label):
            prompt = generate_enhanced_template(label, obj_vocab, subact_vocab, body_part_vocab)
            safe_name = get_safe_filename(label)
            filename = os.path.join(templates_dir, f"{safe_name}_template.txt")
            with open(filename, 'w') as f:
                f.write(prompt)
            return label
        
        # Generate templates in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
            futures = {executor.submit(process_class, label): label for label in class_labels}
            
            # Use tqdm for progress bar
            for future in tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures), 
                desc="Generating templates"
            ):
                results.append(future.result())
        
        # Generate batch template files
        num_batches = (len(class_labels) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(class_labels))
            batch_labels = class_labels[start_idx:end_idx]
            
            # Create vocabulary list strings
            obj_list_str = "\n".join([f"- {obj}" for obj in obj_vocab])
            subact_list_str = "\n".join([f"- {subact}" for subact in subact_vocab])
            body_part_list_str = "\n".join([f"- {part}" for part in body_part_vocab])
            
            batch_prompt = f"""Generate knowledge graphs for the following actions using ONLY terms from the provided object, body part, and subaction vocabularies.

Object vocabulary:
{obj_list_str}

Body part vocabulary:
{body_part_list_str}

Subaction vocabulary:
{subact_list_str}

For each action, please generate:
1. 2-3 relevant objects for this action (select from object vocabulary)
2. 1-2 most important body parts in this action (select from body part vocabulary)
3. 2-3 relevant subactions that make up this action (select from subaction vocabulary)
4. Action-object relation triples describing how the action relates to these objects
5. Action-body part relation triples describing how the action relates to body parts
6. Subaction-subaction relation triples describing temporal relationships between subactions

Output format (YAML):"""
            
            # Add example
            example_label = batch_labels[0]
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
            for label in batch_labels:
                batch_prompt += f"- {label}\n"
            
            batch_filename = os.path.join(output_dir, f"batch_{batch_idx+1}_template.txt")
            with open(batch_filename, 'w') as f:
                f.write(batch_prompt)
        
        return results
    
    # Generate template files
    print("Generating template files...")
    try:
        import concurrent.futures
        from tqdm import tqdm
        templates_results = generate_template_files_parallel(
            class_labels, consolidated_obj_vocab, consolidated_subact_vocab, 
            bp_vocab, output_dir
        )
    except ImportError:
        print("Warning: Cannot import parallel processing modules. Using sequential processing.")
        # Generate templates sequentially
        templates_dir = os.path.join(output_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        for label in class_labels:
            prompt = generate_enhanced_template(label, consolidated_obj_vocab, consolidated_subact_vocab, bp_vocab)
            safe_name = get_safe_filename(label)
            with open(os.path.join(templates_dir, f"{safe_name}_template.txt"), 'w') as f:
                f.write(prompt)
    
    # Save vocabularies
    with open(os.path.join(output_dir, 'object_vocabulary.txt'), 'w') as f:
        for obj in consolidated_obj_vocab:
            f.write(f"{obj}\n")
    
    with open(os.path.join(output_dir, 'subaction_vocabulary.txt'), 'w') as f:
        for subact in consolidated_subact_vocab:
            f.write(f"{subact}\n")
    
    with open(os.path.join(output_dir, 'body_part_vocabulary.txt'), 'w') as f:
        for part in bp_vocab:
            f.write(f"{part}\n")
    
    # Save metadata
    metadata = {
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocabulary_stats": {
            "objects": len(consolidated_obj_vocab),
            "subactions": len(consolidated_subact_vocab),
            "body_parts": len(bp_vocab),
            "similarity_threshold": similarity_threshold
        },
        "class_count": len(class_labels),
        "processing_time": f"{time.time() - start_time:.2f} seconds"
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create sample constrained ASKG
    print("Creating sample constrained ASKG...")
    
    # Create more realistic example function
    def create_realistic_example(action_label, objs, subacts, body_parts, relevant_obj_maps=None, relevant_subact_maps=None):
        """Create a more realistic example ASKG for a specific action"""
        # Select relevant objects and subactions for this specific action
        if relevant_obj_maps and action_label in relevant_obj_maps:
            selected_objs = relevant_obj_maps[action_label][:min(3, len(relevant_obj_maps[action_label]))]
        else:
            # Default to first 3 objects if no mapping is available
            selected_objs = objs[:min(3, len(objs))]
        
        if relevant_subact_maps and action_label in relevant_subact_maps:
            selected_subacts = relevant_subact_maps[action_label][:min(3, len(relevant_subact_maps[action_label]))]
        else:
            # Default to first 3 subactions if no mapping is available
            selected_subacts = subacts[:min(3, len(subacts))]
            
        # Select 1-2 body parts
        selected_body_parts = body_parts[:min(2, len(body_parts))]
        
        # Create action-object triples with more meaningful relations based on action type
        act_obj_triples = []
        for obj in selected_objs:
            relation = "involves"
            
            # Create more specific relations based on action and object patterns
            if "take" in action_label or "put" in action_label:
                relation = "manipulates"
            elif "eat" in action_label or "drink" in action_label:
                relation = "consumes"
            elif "throw" in action_label:
                relation = "propels"
            elif "wear" in action_label or "jacket" in action_label or "hat" in action_label:
                relation = "wears"
            
            act_obj_triples.append(f"<{action_label}, {relation}, {obj}>")
        
        # Create action-body part triples
        act_body_part_triples = []
        for part in selected_body_parts:
            relation = "uses"
            
            # Create more specific relations based on body part
            if part == "head":
                if "nod" in action_label or "shake" in action_label:
                    relation = "moves"
                else:
                    relation = "involves"
            elif part == "upper limbs":
                if "throw" in action_label or "catch" in action_label:
                    relation = "requires"
                else:
                    relation = "uses"
            
            act_body_part_triples.append(f"<{action_label}, {relation}, {part}>")
        
        # Create subaction relation triples with more meaningful temporal relations
        sub_act_rel_triples = []
        if len(selected_subacts) >= 2:
            # Create a linear sequence of subactions
            for i in range(len(selected_subacts) - 1):
                sub_act_rel_triples.append(f"<{selected_subacts[i]}, precedes, {selected_subacts[i+1]}>")
        
        return {
            "label": action_label,
            "obj_li": selected_objs,
            "body_part_li": selected_body_parts,
            "act_obj_triples": act_obj_triples,
            "act_body_part_triples": act_body_part_triples,
            "sub_act_li": selected_subacts,
            "sub_act_rel_triples": sub_act_rel_triples
        }
    
    # Create sample constrained ASKG
    sample_constrained_askg = {}
    
    # Select diverse sample actions (first, middle, last)
    sample_indices = [0, len(class_labels)//2, len(class_labels)-1]
    
    # Try to add some random samples
    try:
        import random
        sample_indices.extend(random.sample(range(1, len(class_labels)-1), min(7, len(class_labels)-2)))
        sample_indices = sorted(list(set(sample_indices)))
    except ImportError:
        # If random module isn't available, add some fixed indices
        if len(class_labels) > 10:
            sample_indices.extend([3, 5, 7, 9])
    
    # Create more realistic examples
    for idx in sample_indices:
        if idx < len(class_labels):
            action_label = class_labels[idx]
            sample_constrained_askg[action_label] = create_realistic_example(
                action_label, 
                consolidated_obj_vocab, 
                consolidated_subact_vocab,
                bp_vocab,
                relevant_objs,
                relevant_subacts
            )
    
    # Save sample constrained ASKG
    sample_dir = os.path.join(output_dir, 'sample_outputs')
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, 'sample_constrained_askg.yml'), 'w') as f:
        yaml.dump(sample_constrained_askg, f, default_flow_style=False)
    
    print(f"Vocabulary constrained templates and sample ASKG saved to {output_dir}")
    print(f"Processing complete in {time.time() - start_time:.2f} seconds")
    
    return consolidated_obj_vocab, consolidated_subact_vocab, bp_vocab, obj_mapping, subact_mapping

def visualize_vocabulary_constraints(obj_vocab, subact_vocab, obj_clusters, subact_clusters, output_dir):
    """Create visualizations of vocabulary constraints"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualize vocabulary size distribution
    plt.figure(figsize=(12, 6))
    
    # Count cluster sizes
    obj_sizes = [len(cluster) for canonical, cluster in obj_clusters.items()]
    subact_sizes = [len(cluster) for canonical, cluster in subact_clusters.items()]
    
    # Plot histograms
    plt.subplot(1, 2, 1)
    plt.hist(obj_sizes, bins=10, alpha=0.7, color='skyblue')
    plt.title('Object Cluster Sizes')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(subact_sizes, bins=10, alpha=0.7, color='lightgreen')
    plt.title('Subaction Cluster Sizes')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'))
    
    # 2. Visualize top clusters
    plt.figure(figsize=(14, 8))
    
    # Sort clusters by size
    top_obj_clusters = sorted([(canonical, len(cluster)) for canonical, cluster in obj_clusters.items()], 
                             key=lambda x: x[1], reverse=True)[:15]
    
    # Plot top object clusters
    plt.subplot(1, 2, 1)
    plt.barh([item[0] for item in top_obj_clusters], [item[1] for item in top_obj_clusters], color='skyblue')
    plt.title('Top 15 Object Clusters by Size')
    plt.xlabel('Cluster Size')
    plt.ylabel('Canonical Object')
    
    # Sort subaction clusters by size
    top_subact_clusters = sorted([(canonical, len(cluster)) for canonical, cluster in subact_clusters.items()], 
                                key=lambda x: x[1], reverse=True)[:15]
    
    # Plot top subaction clusters
    plt.subplot(1, 2, 2)
    plt.barh([item[0] for item in top_subact_clusters], [item[1] for item in top_subact_clusters], color='lightgreen')
    plt.title('Top 15 Subaction Clusters by Size')
    plt.xlabel('Cluster Size')
    plt.ylabel('Canonical Subaction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_clusters.png'))
    
    # 3. Create a visualization of vocabulary reduction
    plt.figure(figsize=(10, 6))
    
    # Calculate total terms vs. canonical terms
    total_objs = sum(len(cluster) for cluster in obj_clusters.values())
    total_subacts = sum(len(cluster) for cluster in subact_clusters.values())
    
    # Create bar chart
    categories = ['Objects', 'Subactions']
    original = [total_objs, total_subacts]
    reduced = [len(obj_vocab), len(subact_vocab)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, original, width, label='Original Terms', color='darkblue')
    plt.bar(x + width/2, reduced, width, label='Consolidated Terms', color='lightblue')
    
    # Add percentage labels
    for i in range(len(categories)):
        reduction_pct = 100 * (1 - reduced[i] / original[i])
        plt.text(i, original[i] + 5, f"{reduction_pct:.1f}% reduction", ha='center')
    
    plt.xticks(x, categories)
    plt.ylabel('Count')
    plt.title('Vocabulary Reduction Through Clustering')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vocabulary_reduction.png'))
    
    # 4. Create example cluster visualization for a few large clusters
    for i, (canonical, cluster) in enumerate(sorted(obj_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:3]):
        if len(cluster) > 1:  # Only visualize non-singleton clusters
            plt.figure(figsize=(10, 4))
            plt.title(f'Object Cluster: "{canonical}"')
            plt.axis('off')
            
            # Create a visualization showing the canonical term and its cluster members
            plt.text(0.5, 0.6, canonical, fontsize=16, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc='skyblue', alpha=0.6))
            
            angle = 0
            radius = 0.35
            n_terms = len(cluster)
            
            for j, term in enumerate(cluster):
                if term != canonical:  # Skip the canonical term
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    plt.text(x, y, term, fontsize=12, ha='center', va='center')
                    
                    # Draw a line from canonical to term
                    plt.plot([0.5, x], [0.6, y], 'k-', alpha=0.3)
                    
                    angle += 2 * np.pi / n_terms
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'obj_cluster_example_{i+1}.png'))
            plt.close()
    
    # Save cluster data as CSV for reference
    obj_cluster_data = []
    for canonical, cluster in obj_clusters.items():
        for term in cluster:
            obj_cluster_data.append({
                'Canonical Form': canonical,
                'Term': term,
                'Type': 'Object'
            })
    
    subact_cluster_data = []
    for canonical, cluster in subact_clusters.items():
        for term in cluster:
            subact_cluster_data.append({
                'Canonical Form': canonical,
                'Term': term,
                'Type': 'Subaction'
            })
    
    # Combine data and save to CSV
    all_data = pd.DataFrame(obj_cluster_data + subact_cluster_data)
    all_data.to_csv(os.path.join(output_dir, 'vocabulary_clusters.csv'), index=False)

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
    parser.add_argument('--task', type=str, 
                       choices=['all', 'tsne', 'cooccurrence', 'body_part', 'enhanced', 'temporal', 'vocabulary', 'constrained'],
                       default='all', help='Specific task to run')
    parser.add_argument('--similarity', type=float, default=0.7,
                        help='Similarity threshold for vocabulary clustering (0.0-1.0)')
    args = parser.parse_args()
    
    # Validate similarity threshold
    if args.similarity < 0 or args.similarity > 1:
        print(f"Warning: Similarity threshold {args.similarity} out of range (0.0-1.0). Using default value 0.7.")
        args.similarity = 0.7
    
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
    
    # Added: body part analysis
    if args.task in ['all', 'body_part']:
        run_body_part_analysis(args.askg, dirs['body_part_matrices'])
    
    # Added: enhanced ASKG generation
    enhanced_askg_file = args.askg
    if args.task in ['all', 'enhanced']:
        enhanced_askg_file = run_enhanced_askg_generation(args.askg, dirs['enhanced_askg'])
    
    if args.task in ['all', 'temporal']:
        run_temporal_relation_analysis(args.askg, dirs['temporal_relations'])
    
    if args.task in ['all', 'vocabulary']:
        run_vocabulary_generation(args.askg, args.labels, dirs['vocab_clustering'])
    
    # Modified: use enhanced ASKG for vocabulary constraints
    if args.task in ['all', 'constrained']:
        # Use enhanced ASKG if available
        askg_for_constrained = enhanced_askg_file if args.task in ['all', 'enhanced'] else args.askg
        run_vocabulary_constrained(askg_for_constrained, args.labels, dirs['vocab_constrained'], args.similarity)
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to {args.output}")

if __name__ == "__main__":
    main()