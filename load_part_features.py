import torch
import numpy as np

part_visual_feature = torch.from_numpy(np.load("save_feature/part_visual_feature.npy"))
part_mu_feature = torch.from_numpy(np.load("save_feature/part_mu_feature.npy"))
part_logvar_feature = torch.from_numpy(np.load("save_feature/part_logvar_feature.npy"))
global_visual_feature = torch.from_numpy(np.load("save_feature/global_visual_feature.npy"))

# 查看各个特征的大小和数据结构
print("part_visual_feature:", part_visual_feature.shape, part_visual_feature.dtype)
print("part_mu_feature:", part_mu_feature.shape, part_mu_feature.dtype)
print("part_logvar_feature:", part_logvar_feature.shape, part_logvar_feature.dtype)
print("global_visual_feature:", global_visual_feature.shape, global_visual_feature.dtype)