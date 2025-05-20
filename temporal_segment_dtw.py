import numpy as np
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from data_cnn60_origin import AverageMeter, NTUDataLoaders

def segment_skeleton_sequences_with_dtw(skeleton_batch, num_segments=list):
    """
    Segment batch skeleton sequences using DTW distance, with specified number of segments
    
    Args:
        skeleton_batch: Batch of skeleton sequences, shape [batch_size, embedding_size, frames, joints]
                        Expected to be a PyTorch tensor
        num_segments: List of number of segments to divide each sequence into
    
    Returns:
        segment_points_batch: List of lists containing segment points for each sample
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(skeleton_batch, torch.Tensor):
        skeleton_batch = skeleton_batch.detach().cpu().numpy()
    
    batch_size, embedding_size, num_frames, num_joints = skeleton_batch.shape
    segment_points_batch = []
    
    for batch_idx in range(batch_size):
        # Get current sample's skeleton sequence
        skeleton_sequence = skeleton_batch[batch_idx]
        
        # Flatten embedding_size dimension to feature vector, shape becomes [frames, embedding_size * joints]
        skeleton_sequence_flat = np.reshape(skeleton_sequence, 
                                           (embedding_size, num_frames, num_joints))
        skeleton_sequence_flat = np.transpose(skeleton_sequence_flat, (1, 0, 2))
        skeleton_sequence_flat = np.reshape(skeleton_sequence_flat, 
                                           (num_frames, embedding_size * num_joints))
        
        # Compute DTW distance matrix between frames
        dtw_matrix = compute_dtw_matrix(skeleton_sequence_flat)
        
        # Perform segmentation based on DTW distances
        segment_points = find_optimal_segments(dtw_matrix, num_segments[batch_idx])
        
        segment_points_batch.append(segment_points)
    
    return segment_points_batch

def compute_dtw_matrix(sequence):
    """
    Compute DTW distance matrix between each pair of frames in the sequence
    
    Args:
        sequence: Sequence with shape [frames, features]
    
    Returns:
        dtw_matrix: DTW distance matrix with shape [frames, frames]
    """
    num_frames = sequence.shape[0]
    dtw_matrix = np.zeros((num_frames, num_frames))
    
    # Use window constraint to reduce computation, only calculate distances near diagonal
    window_size = min(num_frames // 4, 20)  # Adjust window size
    
    for i in range(num_frames):
        # Only compute upper triangular part as DTW distance is symmetric
        for j in range(i, min(i + window_size + 1, num_frames)):
            if i == j:
                dtw_matrix[i, j] = 0
            else:
                # Use fastdtw library for faster computation
                # First window starts from frame i, second from frame j
                window_i = sequence[i:min(i+window_size, num_frames)]
                window_j = sequence[j:min(j+window_size, num_frames)]
                
                distance, _ = fastdtw(window_i, window_j, dist=euclidean)
                dtw_matrix[i, j] = distance
                dtw_matrix[j, i] = distance  # Symmetric matrix
    
    return dtw_matrix

def find_optimal_segments(dtw_matrix, num_segments):
    """
    Find optimal segmentation points based on DTW distance matrix
    
    Args:
        dtw_matrix: DTW distance matrix with shape [frames, frames]
        num_segments: Number of segments to divide sequence into
    
    Returns:
        segment_points: List of segment boundary indices
    """
    num_frames = dtw_matrix.shape[0]
    
    # Method 1: Use cumulative DTW distance curve
    cumulative_dissimilarity = np.zeros(num_frames - 1)
    for i in range(num_frames - 1):
        cumulative_dissimilarity[i] = dtw_matrix[i, i+1]
    
    # Smooth the cumulative distance curve
    window_size = 5
    smoothed_dissimilarity = np.convolve(cumulative_dissimilarity, 
                                         np.ones(window_size)/window_size, 
                                         mode='same')
    
    # Use K-means clustering to find optimal segmentation points
    # First create feature vectors including frame indices and corresponding dissimilarity values
    features = np.column_stack((np.arange(num_frames - 1), smoothed_dissimilarity))
    
    # Give higher weight to dissimilarity for better segmentation
    features[:, 1] = features[:, 1] * 10  # Increase weight of dissimilarity
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_segments - 1, random_state=0).fit(features)
    
    # Get segmentation points
    cluster_centers = kmeans.cluster_centers_
    segment_indices = cluster_centers[:, 0].astype(int)
    segment_indices.sort()  # Ensure segmentation points are ordered
    
    # Make sure segment points are within valid range
    segment_indices = np.clip(segment_indices, 1, num_frames - 2)
    
    return segment_indices.tolist()

def visualize_segmentation(skeleton_sequence, segment_points):
    """
    Visualize segmentation results
    
    Args:
        skeleton_sequence: Skeleton sequence
        segment_points: List of segmentation points
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(skeleton_sequence, torch.Tensor):
        skeleton_sequence = skeleton_sequence.detach().cpu().numpy()
        
    # Flatten skeleton sequence to 2D representation
    flattened_sequence = np.mean(skeleton_sequence, axis=(0, 2))
    
    plt.figure(figsize=(15, 5))
    plt.plot(flattened_sequence)
    
    # Draw segmentation points
    for point in segment_points:
        plt.axvline(x=point, color='r', linestyle='--')
    
    plt.title('Skeleton Sequence Segmentation')
    plt.xlabel('Frames')
    plt.ylabel('Average Joint Position')
    plt.show()

def representative_segs(skeleton_batch, segment_points_batch):
    """
    Obtain representative segments by averaging over the frame dimension.
    Returns all representative segments across all samples as a single tensor.
    
    Args:
        skeleton_batch: Batch of skeleton sequences with shape [batch_size, embedding_size, frames, joints]
                        Expected to be a PyTorch tensor
        segment_points_batch: List of lists containing segment points for each sample
                             Each inner list contains the frame indices for segment boundaries
    
    Returns:
        rep_segs: Single tensor containing all representative segments from all samples
                  Shape: [total_segments, embedding_size, joints]
    """
    # Ensure input is PyTorch tensor
    if not isinstance(skeleton_batch, torch.Tensor):
        skeleton_batch = torch.tensor(skeleton_batch)
    
    batch_size, embedding_size, num_frames, num_joints = skeleton_batch.shape
    
    # List to collect all representative segments from all samples
    all_rep_segs = []
    
    for batch_idx in range(batch_size):
        # Get current sample's skeleton sequence and segment points
        skeleton_sequence = skeleton_batch[batch_idx]
        segment_points = [0] + segment_points_batch[batch_idx] + [num_frames - 1]

        # Add start and end points for complete segmentation
        num_seg = len(segment_points) - 1
        # Process each segment
        for seg_idx in range(num_seg):
            start_frame = segment_points[seg_idx]
            end_frame = segment_points[seg_idx + 1]
            
            # Handle edge case where segments might be identical
            if start_frame == end_frame:
                segment_frames = skeleton_sequence[:, start_frame, :]
            else:
                segment_frames = skeleton_sequence[:, start_frame:end_frame, :]
                rep_seg = torch.mean(segment_frames, dim=1)     # torch.Size([256, 25])
            
            # Add to collection
            all_rep_segs.append(rep_seg)
    
    # Stack all representative segments into a single tensor
    # Shape: [total_segments, embedding_size, joints]
    if all_rep_segs:
        rep_segs = torch.stack(all_rep_segs, dim=0)     # torch.Size([80, 256, 25])
    else:
        # Handle empty case
        rep_segs = torch.zeros(0, embedding_size, num_joints)
    
    return rep_segs

# Example usage
if __name__ == "__main__":
    dataset_path = "/home/penghan/HAR-MLDA/STAR/fl_features/shift_ntu60_5_r"
    batch_size = 4
    ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
    train_loader = ntu_loaders.get_train_loader(batch_size, 0)
    
    for i, (inputs, target) in enumerate(train_loader):
        skeleton_batch = inputs
        segment_points_batch = segment_skeleton_sequences_with_dtw(skeleton_batch, num_segments=3)
    
        # Print results
        for i, points in enumerate(segment_points_batch):
            print(f"Sample {i} segment points: {points}")
        
        # Visualize segmentation for first sample
        visualize_segmentation(skeleton_batch[0], segment_points_batch[0])