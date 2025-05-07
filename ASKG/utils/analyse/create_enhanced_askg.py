import yaml
import os
import re
import argparse
from collections import Counter

# Define body part vocabulary
def get_part_names(body_part=4):
    if body_part == 6:
        part_names = ["head", "hand", "arm", "hip", "leg", "foot"]
    elif body_part == 4:
        part_names = ["head", "upper limbs", "hip", "lower limbs"]
    elif body_part == 2:
        part_names = ["upper body", "lower body"]
    return part_names

# Body part keywords
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

# Identify body parts from objects
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

# Create enhanced ASKG
def create_enhanced_askg(askg_data):
    """Create enhanced ASKG with separated body parts"""
    enhanced_askg = {}
    
    for action_label, action_info in askg_data.items():
        # Create new action info with same structure but added body_part_li
        new_action_info = {
            "label": action_info.get("label", action_label),
            "obj_li": [],
            "body_part_li": [],
            "act_obj_triples": [],
            "act_body_part_triples": [],
            "sub_act_li": action_info.get("sub_act_li", []),
            "sub_act_rel_triples": action_info.get("sub_act_rel_triples", [])
        }
        
        # Get original objects
        obj_li = []
        if 'obj_li' in action_info and action_info['obj_li']:
            for obj in action_info['obj_li']:
                obj_name = obj
                if isinstance(obj, dict) and 'object' in obj:
                    obj_name = obj['object']
                
                # Skip excluded objects
                if obj_name.lower() in EXCLUDED_OBJECTS:
                    continue
                
                obj_li.append(obj_name)
        
        # Identify body parts from objects
        body_parts = identify_body_parts(obj_li)
        
        # Filter out body parts from objects
        filtered_objs = []
        for obj in obj_li:
            is_body_part = False
            for part, keywords in BODY_PART_KEYWORDS.items():
                if any(keyword in obj.lower() for keyword in keywords):
                    is_body_part = True
                    break
            
            if not is_body_part:
                filtered_objs.append(obj)
        
        # Update new action info
        new_action_info["obj_li"] = filtered_objs
        new_action_info["body_part_li"] = body_parts
        
        # Update triples
        if 'act_obj_triples' in action_info and action_info['act_obj_triples']:
            for triple in action_info['act_obj_triples']:
                triple_str = str(triple)
                # Use regex to extract parts
                match = re.search(r'<([^,]+),\s*([^,]+),\s*([^>]+)>', triple_str)
                
                if match:
                    subj, rel, obj = match.groups()
                    subj = subj.strip()
                    rel = rel.strip()
                    obj = obj.strip()
                    
                    # Check if object is a body part
                    is_body_part = False
                    for part, keywords in BODY_PART_KEYWORDS.items():
                        if any(keyword in obj.lower() for keyword in keywords):
                            is_body_part = True
                            # Create body part triple
                            new_triple = f"<{subj}, {rel}, {part}>"
                            if new_triple not in new_action_info["act_body_part_triples"]:
                                new_action_info["act_body_part_triples"].append(new_triple)
                            break
                    
                    # If not a body part and not excluded, keep original triple
                    if not is_body_part and obj.lower() not in EXCLUDED_OBJECTS:
                        new_action_info["act_obj_triples"].append(triple)
        
        # Keep standard subaction structure
        if 'sub_act_li' in action_info:
            new_action_info["sub_act_li"] = action_info["sub_act_li"]
        
        if 'sub_act_rel_triples' in action_info:
            new_action_info["sub_act_rel_triples"] = action_info["sub_act_rel_triples"]
        elif 'act_rel_triples' in action_info:
            new_action_info["sub_act_rel_triples"] = action_info["act_rel_triples"]
        
        # Add to enhanced ASKG
        enhanced_askg[action_label] = new_action_info
    
    return enhanced_askg

# Create separate body parts YAML file
def create_body_parts_yaml(enhanced_askg, output_file):
    """Create separate YAML file with only body part data"""
    body_parts_data = {}
    
    for action_label, action_info in enhanced_askg.items():
        body_parts_data[action_label] = {
            "label": action_info.get("label", action_label),
            "body_part_li": action_info.get("body_part_li", []),
            "act_body_part_triples": action_info.get("act_body_part_triples", [])
        }
    
    # Save to YAML
    with open(output_file, 'w') as f:
        yaml.dump(body_parts_data, f, default_flow_style=False)
    
    return body_parts_data

# Main function
def main(askg_file, output_dir):
    """Main function for enhancing ASKG"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ASKG data
    askg_data = load_askg_data(askg_file)
    print(f"Loaded ASKG with {len(askg_data)} actions")
    
    # Create enhanced ASKG
    enhanced_askg = create_enhanced_askg(askg_data)
    print(f"Created enhanced ASKG with {len(enhanced_askg)} actions")
    
    # Save enhanced ASKG
    enhanced_askg_file = os.path.join(output_dir, "enhanced_askg.yml")
    with open(enhanced_askg_file, 'w') as f:
        yaml.dump(enhanced_askg, f, default_flow_style=False)
    print(f"Enhanced ASKG saved to {enhanced_askg_file}")
    
    # Create and save body parts YAML
    body_parts_file = os.path.join(output_dir, "body_parts.yml")
    create_body_parts_yaml(enhanced_askg, body_parts_file)
    print(f"Body parts data saved to {body_parts_file}")
    
    # Print statistics
    original_obj_count = 0
    enhanced_obj_count = 0
    body_part_count = 0
    
    for action_label, action_info in askg_data.items():
        if 'obj_li' in action_info and action_info['obj_li']:
            original_obj_count += len(action_info['obj_li'])
    
    for action_label, action_info in enhanced_askg.items():
        if 'obj_li' in action_info and action_info['obj_li']:
            enhanced_obj_count += len(action_info['obj_li'])
        if 'body_part_li' in action_info and action_info['body_part_li']:
            body_part_count += len(action_info['body_part_li'])
    
    print("\nEnhanced ASKG Statistics:")
    print(f"  Total original objects: {original_obj_count}")
    print(f"  Total enhanced objects: {enhanced_obj_count}")
    print(f"  Total body parts: {body_part_count}")

# If executed as script
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Enhance ASKG with body parts')
    parser.add_argument('--askg', type=str, default='classes_ASKG_ntu_checked.yml',
                       help='Path to ASKG data file')
    parser.add_argument('--output', type=str, default='enhanced_askg_output',
                       help='Output directory')
    args = parser.parse_args()
    
    main(args.askg, args.output)