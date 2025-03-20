import os
import numpy as np
import ipdb

folder_path = "/usr1/home/s124mdg53_04/Dissertation/resources/sk_feats/shift_ntu60_val_5_r"  # 替换为实际路径
# ipdb.set_trace()
for file_name in os.listdir(folder_path):
    if file_name.endswith(".npy"):  # 只处理 .npy 文件\n",
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path, allow_pickle=True)  # 加载 .npy 文件\n",
        print(f"====={file_name}: {data.shape}=====")  # 打印文件名和形状\n",
        if 'label' in file_name:
            print(data[0:10])

# data_path = "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict/train.npy"
# data = np.load(data_path, allow_pickle=True).item()
# print(data['global'].shape, data['part'].shape)