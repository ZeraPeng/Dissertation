import os, time
import openai
import yaml
from dotenv import load_dotenv
import sys

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, 
                                 openai.error.RateLimitError, openai.error.ServiceUnavailableError, 
                                 openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    """
    Call OpenAI API with exponential backoff retry logic
    """
    return openai.ChatCompletion.create(**kwargs)

# Adjust the paths as needed
data_path = '/Users/fgg/Desktop/code/ASKG/utils/'  # Update this path to your actual path
# read file path
label_ASKG_path = os.path.join(data_path, 'action_knowledge_graph.yaml')
# write file path
xprompt_path = os.path.join(data_path, 'prompttest.yml')

print(f"Reading data from: {label_ASKG_path}")
print(f"Output will be written to: {xprompt_path}")

try:
    with open(label_ASKG_path, 'r') as label_ASKG_file:
        data = yaml.safe_load(label_ASKG_file)
        print(f"Successfully loaded data from {label_ASKG_path}")
except Exception as e:
    print(f"Error loading data file: {e}")
    sys.exit(1)

print(f"Data type: {type(data)}")
if isinstance(data, dict):
    print(f"Data contains {len(data)} items")
    print("Action labels found:")
    for i, label in enumerate(data.keys()):
        print(f"  {i+1}. {label}")
else:
    print(f"Data is not a dictionary, but a {type(data)}")
    sys.exit(1)

# continue from last run
last = 0

comments_KG_text = {
    'label': 'action label',
    'act_obj_triples': 'action-object relation triples',
    'act_rel_triples': 'action-action relation triples',
}

# one-shot prompt
one_shot_label = 'drinking water'
one_shot_user = {
      "role": "user",
      "content": "action label: drinking water\n\naction-object relation triples:\n1 - <drinking water, involves, water>\n2 - <drinking water, uses, cup>\n3 - <drinking water, where water, in cup>\n\naction-action relation triples:\n1 - <drinking water, starts with, grasping>\n2 - <grasping, precedes, lifting>\n3 - <lifting, precedes, tilting>\n4 - <tilting, precedes, sipping>\n5 - <sipping, precedes, swallowing>"
    }
one_shot_assistant = {
      "role": "assistant",
      "content": "drinking water:\n  label: drinking water\n  xprompt_ao:\n    - which involves water\n    - which uses a cup\n    - where water is in a cup\n  xprompt_aa:\n    - starting with grasping\n    - where grasping precedes lifting\n    - where lifting precedes tilting\n    - where tilting precedes sipping\n    - where sipping precedes swallowing"
    }

# Initialize the list to store all action dictionaries
all_actions = []

try:
    with open(xprompt_path, 'w') as xprompt_file:
        # Process each action label
        for no, (label, info) in enumerate(data.items()):
            if no < last:
                print(f"Skipping item {no+1}: {label} (already processed)")
                continue
                
            print(f"\nProcessing item {no+1}: {label}")
            
            user_input = f'action label: {label}\n\n'
            
            # Handle different data structures that might be present
            if isinstance(info, dict):
                for k, v in info.items():
                    if k in comments_KG_text.keys():
                        key = comments_KG_text[k]
                        if isinstance(v, str) or v is None:
                            user_input += f'{key}: {v}\n\n'
                        elif isinstance(v, list):
                            user_input += f'{key}:\n'
                            for i, vi in enumerate(v):
                                user_input += f'{i+1} - {vi}\n'
                            user_input += '\n'
                        else:
                            user_input += f'{key}: \n\n'
            else:
                # In case info is not a dictionary (like a string or None)
                print(f"Warning: Data for '{label}' is not in expected format. Type: {type(info)}")
                continue  # Skip this entry
            
            print(f"Prepared user input for {label}")
            
            try:
                start_time = time.time()
                print(f"Calling OpenAI API for {label}...")
                response = chat_completion_with_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a commonsense knowledge base, especially for human actions.
    You will be provided with relation triples related to the action label below.
    Use the following step-by-step instructions to respond to user inputs:

    1. For each action-object relation triple, create a prompt that describes how objects interact with the main action label.
       - Avoid references to human body parts (like mouth, hands, etc.) unless absolutely essential.
       - Focus on the interaction between objects and the main action, not sub-actions.

    2. For each action-action relation triple, create a prompt that describes the sequence or relationship between sub-actions.

    3. Output the final answers in YAML format with only these fields:
       - label: the action label
       - xprompt_ao: action-object relation prompts (list)
       - xprompt_aa: action-action relation prompts (list)

    Keep your responses clear and concise."""
                        },
                        one_shot_user,
                        one_shot_assistant,
                        {
                            "role": "user",
                            "content": user_input.strip()
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                ans_time = time.time()
                consume_time = ans_time - start_time
                content = response.choices[0]["message"]["content"].strip()
                
                print(f"Received response from API for {label}")
                
                # Parse the YAML response
                try:
                    action_dict = yaml.safe_load(content)
                    print(f"Successfully parsed YAML for {label}")
                    
                    # Add to the list of actions
                    if isinstance(action_dict, dict):
                        all_actions.append(action_dict)
                        print(f"Added {label} to actions list")
                    else:
                        print(f"Warning: Expected a dictionary for {label} but got {type(action_dict)}")
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML for {label}: {e}")
                    print(f"Raw content: {content}")
                
                print(content)
                print(f"##No.{no+1} time consuming : {consume_time:.3f} s##")
                
                # Optional: Write intermediate results to file
                try:
                    with open(xprompt_path + '.temp', 'w') as temp_file:
                        yaml.dump(all_actions, temp_file, default_flow_style=False, allow_unicode=True)
                    print(f"Saved intermediate results after processing {label}")
                except Exception as e:
                    print(f"Warning: Could not save intermediate results: {e}")
                
            except Exception as api_error:
                print(f"Error calling OpenAI API for {label}: {api_error}")
        
        # Write all actions to YAML file
        yaml.dump(all_actions, xprompt_file, default_flow_style=False, allow_unicode=True)
        print(f"\nAll actions written to {xprompt_path}")
        print(f"Total number of actions processed: {len(all_actions)}")

except Exception as e:
    print(f"Unhandled error: {e}")
    # Save what we have so far
    try:
        with open(xprompt_path + '.backup', 'w') as backup_file:
            yaml.dump(all_actions, backup_file, default_flow_style=False, allow_unicode=True)
        print(f"Saved backup of processed actions to {xprompt_path}.backup")
    except:
        print("Could not save backup file")