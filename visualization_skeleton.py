import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from data_cnn60_origin import AverageMeter, NTUDataLoaders

def visualize_skeleton(skeleton_data, sample_idx=0, frame_idx=None):
    """
    可视化NTU RGB+D骨架数据
    
    参数:
    - skeleton_data: 形状为[batch_size, embedding_size, num_frames, num_joints]的张量
    - sample_idx: 批次中的样本索引
    - frame_idx: 指定要可视化的帧，如果为None则创建动画
    """
    # NTU RGB+D数据集的骨骼连接
    connections = [
        (0, 1), (1, 20), (20, 2), (2, 3), # 脊柱
        (20, 4), (4, 5), (5, 6), (6, 7), # 右臂
        (20, 8), (8, 9), (9, 10), (10, 11), # 左臂
        (0, 12), (12, 13), (13, 14), (14, 15), # 右腿
        (0, 16), (16, 17), (17, 18), (18, 19), # 左腿
        (20, 22), (22, 23), (23, 24), # 右手指
        (20, 21) # 左手指
    ]
    
    # 假设前3个维度是关节的x,y,z坐标
    # 从tensor中提取单个样本
    sample = skeleton_data[sample_idx]  # [embedding_size, num_frames, num_joints]
    
    # 提取坐标信息 - 假设前3个嵌入维度是x,y,z坐标
    # 或者，如果嵌入不是直接的坐标，可能需要额外处理
    coordinates = sample[:3, :, :]  # [3, num_frames, num_joints]
    
    # 调整到NumPy格式 [num_frames, num_joints, 3]
    coordinates = coordinates.permute(1, 2, 0).cpu().numpy()
    
    # 创建图形窗口
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if frame_idx is not None:
        # 可视化单个帧
        frame = coordinates[frame_idx]
        
        # 绘制骨骼点
        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='blue', marker='o', s=50)
        
        # 绘制骨骼连接
        for connection in connections:
            ax.plot([frame[connection[0], 0], frame[connection[1], 0]],
                   [frame[connection[0], 1], frame[connection[1], 1]],
                   [frame[connection[0], 2], frame[connection[1], 2]], c='red')
        
        ax.set_title(f'Skeleton Visualization (Frame {frame_idx})')
    else:
        # 创建动画
        def update(frame_num):
            ax.clear()
            frame = coordinates[frame_num]
            
            # 绘制骨骼点
            ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c='blue', marker='o', s=50)
            
            # 绘制骨骼连接
            for connection in connections:
                ax.plot([frame[connection[0], 0], frame[connection[1], 0]],
                       [frame[connection[0], 1], frame[connection[1], 1]],
                       [frame[connection[0], 2], frame[connection[1], 2]], c='red')
            
            ax.set_title(f'Skeleton Visualization (Frame {frame_num})')
            ax.set_xlim([np.min(coordinates[:, :, 0]), np.max(coordinates[:, :, 0])])
            ax.set_ylim([np.min(coordinates[:, :, 1]), np.max(coordinates[:, :, 1])])
            ax.set_zlim([np.min(coordinates[:, :, 2]), np.max(coordinates[:, :, 2])])
            
        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=len(coordinates), interval=100)
        
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def pool_joints(skeleton_data, method='mean'):
    """
    池化关节维度
    
    参数:
    - skeleton_data: 形状为[batch_size, embedding_size, num_frames, num_joints]的张量
    - method: 池化方法，可选 'mean'(平均), 'max'(最大), 'weighted'(加权平均), 'group'(分组池化)
    
    返回:
    - pooled_data: 形状为[batch_size, embedding_size, num_frames]的张量
    """
    batch_size, embedding_size, num_frames, num_joints = skeleton_data.shape
    
    if method == 'mean':
        # 平均池化
        return torch.mean(skeleton_data, dim=3)
    
    elif method == 'max':
        # 最大池化
        return torch.max(skeleton_data, dim=3)[0]
    
    elif method == 'group':
        # 基于关节组的池化
        # 定义关节组，例如躯干、左臂、右臂、左腿、右腿
        groups = {
            'spine': [0, 1, 20, 2, 3],
            'right_arm': [20, 4, 5, 6, 7, 22, 23, 24],
            'left_arm': [20, 8, 9, 10, 11, 21],
            'right_leg': [0, 12, 13, 14, 15],
            'left_leg': [0, 16, 17, 18, 19]
        }
        
        # 创建结果张量
        pooled_data = torch.zeros(batch_size, embedding_size, num_frames, len(groups), device=skeleton_data.device)
        
        # 对每个组进行池化
        for i, (group_name, joint_indices) in enumerate(groups.items()):
            group_data = skeleton_data[:, :, :, joint_indices]
            pooled_data[:, :, :, i] = torch.mean(group_data, dim=3)
        
        # 如果想进一步压缩到[batch_size, embedding_size, num_frames]
        # 可以对分组后的数据再次进行池化
        return torch.mean(pooled_data, dim=3)
    
    else:
        raise ValueError(f"Unknown pooling method: {method}")

def visualize_pooled_skeleton(pooled_data, sample_idx=0, method_name='mean'):
    """
    可视化池化后的骨架数据
    
    参数:
    - pooled_data: 形状为[batch_size, embedding_size, num_frames]的张量
    - sample_idx: 批次中的样本索引
    - method_name: 池化方法名称，用于标题
    """
    # 从tensor中提取单个样本
    sample = pooled_data[sample_idx]  # [embedding_size, num_frames]
    
    # 假设嵌入的前几个维度可能包含最重要的信息
    # 我们可以选择前几个维度来可视化
    num_features_to_show = min(10, sample.shape[0])  # 显示前10个特征
    features = sample[:num_features_to_show, :].cpu().numpy()  # [num_features_to_show, num_frames]
    
    # 创建图形窗口
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # 1. 特征随时间变化的线图
    ax1 = axes[0]
    frames = np.arange(features.shape[1])
    
    for i in range(features.shape[0]):
        ax1.plot(frames, features[i], label=f'Feature {i+1}')
    
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Feature Value')
    ax1.set_title(f'Feature Evolution Over Time ({method_name} pooling)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # 2. 特征热力图
    ax2 = axes[1]
    im = ax2.imshow(features, aspect='auto', cmap='viridis', 
                   interpolation='nearest', origin='lower')
    ax2.set_xlabel('Frames')
    ax2.set_ylabel('Feature Index')
    ax2.set_title(f'Feature Heatmap ({method_name} pooling)')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax2, label='Feature Value')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


dataset_path = "/home/penghan/HAR-MLDA/STAR/fl_features/shift_ntu60_5_r"
batch_size = 4
ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
train_loader = ntu_loaders.get_val_loader(batch_size, 0)

for i, (inputs, target) in enumerate(train_loader):
    fig, ax = visualize_skeleton(inputs)
    mean_pool = pool_joints(inputs)
    fig_mean, ax = visualize_pooled_skeleton()


    break
    # max_pool = pool_joints(inputs, method='max')
    # visualize_skeleton(max_pool)

    # mean_pool = pool_joints(inputs, method='mean')
    # visualize_skeleton(mean_pool)

    # group_pool = pool_joints(inputs, method='group')
    # visualize_skeleton(group_pool)
