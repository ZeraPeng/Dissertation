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
import json

askg_map_path = 'ASKG/data/ntu/askg_mappings.json'
out_path = 'ASKG/data/vocab/aug/sub_act_vocab_feats_ntu.tar'

ipdb.set_trace()
with open('ASKG/data/ntu/askg_mappings.json', 'r') as f:
    askg_mapping = json.load(f)

vocab = askg_mapping['idx2sa']

num_vocab = len(vocab)
templates = "a human action of"

tokenized_list = []
vocab_list = []
text_list = []
for key, value in vocab.items():
    vocab_list.append(f"{templates} {value}")

for t in vocab_list:
    tokenized_list.append(clip.tokenize(t))

vocab_tokenized = torch.cat(tokenized_list, dim=0)       # [154, 77]

# initialize clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)

vocab_tokenized = vocab_tokenized.to(device)
features = clip_model.encode_text(vocab_tokenized)  
features = features.to('cpu')   # torch.Size([154, 512])

torch.save(features, out_path)