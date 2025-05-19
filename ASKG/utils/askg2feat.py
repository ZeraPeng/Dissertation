import os
import yaml
import pandas as pd
import numpy as np
import clip
import torch
import torch.nn.functional as F
import nltk
from nltk.stem import WordNetLemmatizer
from einops import rearrange  # Added missing import
import ipdb

def get_templates(dataset_name):
    label_file = f"ASKG/data/{dataset_name}/classes_label_{dataset_name}.yml"
    with open(label_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def get_askg(dataset_name):
    askg_file = f"ASKG/data/ntu/classes_ASKG_vocab_{dataset_name}.yml"
    with open(askg_file, 'r') as f:
        askg = yaml.load(f, Loader=yaml.FullLoader)
    return askg

def aug_subaction_prepare_no_expand(data, dataset: str, num_templates: int, cls_prompt_type: str):
    text = get_askg(dataset)
    classes = data # ["c1", "c2", ...]
    num_classes = len(classes)
    n_prompts = [0, 1]
    
    templates = ["a human action of"]
    total_templates = len(templates)
    num_templates = min(num_templates, total_templates)

    tokenized_dict = {}
    cls_text_dict = {}
    text_dict = {}
    sub_act_dict = {} # {0: [xprompt_ao,...], 1: [xprompt_ao], ...}
    obj_dict = {} # {0: [xprompt_aa,...], 1: [xprompt_aa], ...}
    for i, t in enumerate(text):
        sub_act_dict[i] = text[t]['sub_act_li']
        obj_dict[i] = text[t]['obj_li']
    for ii, txt in enumerate(templates):
        text_dict[ii] = {
            'aug': txt,
            'sub_act': [],
            'obj': []
        }
        tokenized_dict[ii] = []
        for i, c in enumerate(classes):
            # ci_xao_list = text_dict[ii]['a'][i][:]
            # ci_xaa_list = text_dict[ii]['a'][i][:]
            ci_sub_act_list = []
            ci_obj_list = []
            for j, t in enumerate(sub_act_dict[i]):
                ci_sub_act_list.append(f"{txt.format(c)} {t}")
            text_dict[ii]['sub_act'].append(ci_sub_act_list)
            for j, t in enumerate(obj_dict[i]):
                ci_obj_list.append(f"{txt.format(c)} {t}")
            text_dict[ii]['obj'].append(ci_obj_list)
        if cls_prompt_type == 'sub_act':
            cls_text_dict[ii] = text_dict[ii]['sub_act']
        elif cls_prompt_type == 'obj':
            cls_text_dict[ii] = text_dict[ii]['obj']
        elif cls_prompt_type == 'mix':
            cls_text_dict[ii] = []
            for i in range(len(classes)):
                cls_text_dict[ii].append(list(set(text_dict[ii]['sub_act'][i] + text_dict[ii]['obj'][i])))
        elif cls_prompt_type == 'pair':
            cls_text_dict[ii] = []
            for i in range(len(classes)):
                cls_text_dict[ii].append(list(set(text_dict[ii]['sub_act'][i] + text_dict[ii]['obj'][i])))
        else:
            cls_text_dict[ii] = text_dict[ii]['aug']
        for t in cls_text_dict[ii]:
            tokenized_item = []
            for n in range(len(t)):
                tokenized_item.append(clip.tokenize(t[n]))
            tokenized_dict[ii].append(torch.cat(tokenized_item))
        # tokenized_dict[0] shape: (120, *, ) -> (number of prompts, classes, token)

        # tokenized_dict[ii] = torch.cat(tokenized_dict[ii])  # (360, 77)
    # cls_tokenized = torch.cat([v for v in tokenized_dict.values()]) # (num_templates max_prompt num_cls) 77
    cls_tokenized = tokenized_dict[0]
    # tensor_list = tokenized_dict[0]
    # max_x = max(tensor.shape[0] for tensor in tensor_list)

    # padded_tensors = []
    # for tensor in tensor_list:
    #     pad_size = max_x - tensor.shape[0]
        
    #     if pad_size > 0:
    #         padded_tensor = F.pad(tensor, (0, 0, 0, pad_size), "constant", 0)
    #     else:
    #         padded_tensor = tensor
            
    #     padded_tensors.append(padded_tensor)

    # cls_tokenized = torch.cat(padded_tensors)

    return cls_tokenized, cls_text_dict, tokenized_dict, num_templates, n_prompts

def aug_feat_processor_sub_action_no_expand(cls_prompt_type='sub_act'):    # Create a class for dataset configuration
    class Config:
        def __init__(self):
            self.data = type('', (), {})()
            self.data.dataset = 'ntu'
            self.data.num_templates = 1

    config = Config()
    # Load dataset
    dataset_name = config.data.dataset
    data = get_templates(dataset_name)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    # Process text features
    classes_feats_file = f"ASKG/data/vocab/{cls_prompt_type}_text_feats_askg_ntu.tar"
    cls_tokenized, cls_text_dict, text_dict, n_templates, n_prompts = aug_subaction_prepare_no_expand(data, config.data.dataset, num_templates=config.data.num_templates, cls_prompt_type=cls_prompt_type)
    # Save cls_text_dict to a YAML file
    cls_text_dict_file = f"ASKG/data/vocab/{cls_prompt_type}_text_dict_askg_ntu.yml"
    with open(cls_text_dict_file, 'w') as f:
        yaml.dump(cls_text_dict, f)
    ipdb.set_trace()

    # Calculate number of classes and text augmentations
    n_classes = int(120)
    num_text_aug = n_prompts[1] * n_templates
    
    # # Rearrange tensor dimensions for processing
    # cls_tokenized = rearrange(cls_tokenized, '(x a) d -> x a d', a=n_classes)      # [3, 120, 77]
    # x, a, d = cls_tokenized.size()
    
    # Encode text features with CLIP
    clip_model.eval()
    with torch.no_grad():
        # Process each batch and ensure it's on the correct device
        classes_features = []
        for i, cls_sub_acts in enumerate(cls_tokenized):
            cls_sub_acts = cls_sub_acts.to(device)
            feature = clip_model.encode_text(cls_sub_acts)  # torch.Size([3, 512])
            classes_features.append(feature)
        # zero-padding
        tensor_list = classes_features
        max_x = max(tensor.shape[0] for tensor in tensor_list)

        padded_tensors = []
        for tensor in tensor_list:
            pad_size = max_x - tensor.shape[0]
            
            if pad_size > 0:
                padded_tensor = F.pad(tensor, (0, 0, 0, pad_size), "constant", 0)
            else:
                padded_tensor = tensor
                
            padded_tensors.append(padded_tensor)

        classes_features = torch.cat(padded_tensors)
        classes_features = rearrange(classes_features, '(x a) d -> x a d', x=n_classes)
        # Move to CPU for saving
        classes_features = classes_features.to('cpu')       # [120, 3, 512]
        torch.save(classes_features, classes_feats_file)


if __name__ == '__main__':
    aug_feat_processor_sub_action_no_expand('sub_act')