import os
import yaml
import pandas as pd
import numpy as np
import clip
import torch
import torch.nn.functional as F
import nltk
from nltk.stem import WordNetLemmatizer
from einops import rearrange 
import ipdb
import json

def get_askg(dataset_name):
    askg_file = f"ASKG/data/ntu/classes_ASKG_vocab_{dataset_name}.yml"
    with open(askg_file, 'r') as f:
        askg = yaml.load(f, Loader=yaml.FullLoader)
    return askg

def get_dataset(dataset_name):
    label_file = f"ASKG/data/{dataset_name}/classes_label_{dataset_name}.yml"
    with open(label_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def subact_get_all_vocab():
    text = get_askg('ntu')
    sub_act_list = []
    obj_list = []
    for i, t in enumerate(text):
        class_label = i
        for sub_act in text[t]['sub_act_li']:
            if sub_act not in sub_act_list:
                sub_act_list.append(sub_act)
        for obj in text[t]['obj_li']:
            if obj not in obj_list:
                obj_list.append(obj)
 
    idx2subact = {i: sub_act for i, sub_act in enumerate(sub_act_list)}
    idx2obj = {i: obj for i, obj in enumerate(obj_list)}

    return idx2subact, idx2obj
    
def classlabels2subact(idx2subact, idx2obj):
    text = get_askg('ntu')
    label = get_dataset('ntu') 
    
    class2subact_idx = {}
    class2obj_idx = {}
    
    subact2idx = {v: k for k, v in idx2subact.items()}
    obj2idx = {v: k for k, v in idx2obj.items()}
    
    for i, t in enumerate(text):
        class_label = i
        
        sub_act_indices = []
        for sub_act in text[t]['sub_act_li']:
            if sub_act in subact2idx:
                sub_act_indices.append(subact2idx[sub_act])
        
        obj_indices = []
        for obj in text[t]['obj_li']:
            if obj in obj2idx:
                obj_indices.append(obj2idx[obj])
        
        class2subact_idx[class_label] = sub_act_indices
        class2obj_idx[class_label] = obj_indices
    
    return class2subact_idx, class2obj_idx

def main(di_path):
    os.makedirs(di_path, exist_ok=True)
    
    idx2subact, idx2obj = subact_get_all_vocab()
    
    class2subact_idx, class2obj_idx = classlabels2subact(idx2subact, idx2obj)
    
    data_to_save = {
        "idx2subact": {str(k): v for k, v in idx2subact.items()},  
        "idx2obj": {str(k): v for k, v in idx2obj.items()},  
        "class2subact_idx": {str(k): v for k, v in class2subact_idx.items()},
        "class2obj_idx": {str(k): v for k, v in class2obj_idx.items()}
    }
    
    json_file_path = os.path.join(di_path, "askg_mappings.json")
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    
    return json_file_path

if __name__ == "__main__":
    di_path = 'ASKG/vocab_idx'
    result_path = main(di_path)
    print(f"Saved to {result_path}")