import numpy as np
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_cnn60_origin import AverageMeter, NTUDataLoaders

def segment_skeleton_sequences_with_dtw(skeleton_batch, num_segments=None, range_seg=None):
    """
    Segment batch skeleton sequences using DTW distance, with specified number of segments or adaptive range
    
    Args:
        skeleton_batch: Batch of skeleton sequences, shape [batch_size, embedding_size, frames, joints]
                        Expected to be a PyTorch tensor
        num_segments: List of number of segments to divide each sequence into (original functionality)
        range_seg: List [min_seg, max_seg] to find optimal number of segments adaptively
    
    Returns:
        segment_points_batch: List of lists containing segment points for each sample
        optimal_segments_batch: List of optimal segment numbers for each sample (only when using range_seg)
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(skeleton_batch, torch.Tensor):
        skeleton_batch = skeleton_batch.detach().cpu().numpy()
    
    batch_size, embedding_size, num_frames = skeleton_batch.shape
    segment_points_batch = []
    optimal_segments_batch = []
    
    for batch_idx in range(batch_size):
        # Get current sample's skeleton sequence
        skeleton_sequence = skeleton_batch[batch_idx]
        
        # Flatten embedding_size dimension to feature vector, shape becomes [frames, embedding_size * joints]
        skeleton_sequence_flat = np.transpose(skeleton_sequence, (1, 0))
        
        # Compute DTW distance matrix between frames
        dtw_matrix = compute_dtw_matrix(skeleton_sequence_flat)
        
        if range_seg is not None:
            # Adaptive segmentation: find optimal number of segments
            optimal_num_segments = find_optimal_num_segments(dtw_matrix, range_seg)
            optimal_segments_batch.append(optimal_num_segments)
            segment_points = find_optimal_segments(dtw_matrix, optimal_num_segments)
        else:
            # Original functionality: use specified number of segments
            segment_points = find_optimal_segments(dtw_matrix, num_segments[batch_idx])
        
        segment_points_batch.append(segment_points)
    
    if range_seg is not None:
        return segment_points_batch, optimal_segments_batch
    else:
        return segment_points_batch

def find_optimal_num_segments(dtw_matrix, range_seg):
    """
    Find optimal number of segments using multiple criteria within the specified range
    
    Args:
        dtw_matrix: DTW distance matrix with shape [frames, frames]
        range_seg: List [min_seg, max_seg] defining the search range
    
    Returns:
        optimal_num_segments: Optimal number of segments
    """
    min_seg, max_seg = range_seg
    num_frames = dtw_matrix.shape[0]
    
    # Ensure valid range
    min_seg = max(2, min_seg)
    max_seg = min(num_frames // 2, max_seg)
    
    if min_seg > max_seg:
        return min_seg
    
    # Prepare features for clustering evaluation
    cumulative_dissimilarity = np.zeros(num_frames - 1)
    for i in range(num_frames - 1):
        cumulative_dissimilarity[i] = dtw_matrix[i, i+1]
    
    # Smooth the cumulative distance curve
    window_size = 5
    smoothed_dissimilarity = np.convolve(cumulative_dissimilarity, 
                                         np.ones(window_size)/window_size, 
                                         mode='same')
    
    features = np.column_stack((np.arange(num_frames - 1), smoothed_dissimilarity))
    features[:, 1] = features[:, 1] * 10  # Increase weight of dissimilarity
    
    # Evaluate different numbers of segments
    scores = {}
    
    for num_seg in range(min_seg, max_seg + 1):
        try:
            # Silhouette score for clustering quality
            kmeans = KMeans(n_clusters=num_seg - 1, random_state=0, n_init=10).fit(features)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette = silhouette_score(features, kmeans.labels_)
            else:
                silhouette = -1
            
            # Inertia (within-cluster sum of squares) - lower is better
            inertia = kmeans.inertia_
            
            # Normalized inertia to make it comparable across different numbers of segments
            normalized_inertia = inertia / (num_seg - 1)
            
            # Segment variance score - measure how well segments are separated
            segment_variance_score = calculate_segment_variance_score(dtw_matrix, num_seg)
            
            # Combined score: balance between clustering quality and segment separation
            # Higher silhouette score is better, lower normalized inertia is better
            # Higher segment variance score is better
            combined_score = (silhouette * 0.4 + 
                            (1 / (1 + normalized_inertia)) * 0.3 + 
                            segment_variance_score * 0.3)
            
            scores[num_seg] = {
                'silhouette': silhouette,
                'normalized_inertia': normalized_inertia,
                'segment_variance': segment_variance_score,
                'combined': combined_score
            }
            
        except Exception as e:
            # Handle edge cases
            scores[num_seg] = {
                'silhouette': -1,
                'normalized_inertia': float('inf'),
                'segment_variance': 0,
                'combined': -1
            }
    
    # Find optimal number of segments based on combined score
    if scores:
        optimal_num_segments = max(scores.keys(), key=lambda k: scores[k]['combined'])
    else:
        optimal_num_segments = min_seg
    
    return optimal_num_segments

def calculate_segment_variance_score(dtw_matrix, num_segments):
    """
    Calculate a score based on how well segments are separated in terms of DTW distances
    
    Args:
        dtw_matrix: DTW distance matrix
        num_segments: Number of segments to evaluate
    
    Returns:
        variance_score: Score indicating segment separation quality
    """
    num_frames = dtw_matrix.shape[0]
    
    # Create equally spaced segments for evaluation
    segment_boundaries = np.linspace(0, num_frames, num_segments + 1, dtype=int)
    
    within_segment_distances = []
    between_segment_distances = []
    
    # Calculate within-segment and between-segment distances
    for i in range(num_segments):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i + 1]
        
        # Within-segment distances
        if end_idx - start_idx > 1:
            segment_dtw = dtw_matrix[start_idx:end_idx, start_idx:end_idx]
            within_distances = segment_dtw[np.triu_indices_from(segment_dtw, k=1)]
            within_segment_distances.extend(within_distances)
        
        # Between-segment distances (with next segment)
        if i < num_segments - 1:
            next_start = segment_boundaries[i + 1]
            next_end = segment_boundaries[i + 2]
            between_distances = dtw_matrix[start_idx:end_idx, next_start:next_end].flatten()
            between_segment_distances.extend(between_distances)
    
    if len(within_segment_distances) == 0 or len(between_segment_distances) == 0:
        return 0
    
    # Good segmentation should have low within-segment distances and high between-segment distances
    mean_within = np.mean(within_segment_distances)
    mean_between = np.mean(between_segment_distances)
    
    if mean_within == 0:
        return 1 if mean_between > 0 else 0
    
    # Variance score: ratio of between-segment to within-segment distances
    variance_score = mean_between / (mean_within + 1e-8)
    
    # Normalize to [0, 1] range
    variance_score = 1 / (1 + np.exp(-variance_score))
    
    return variance_score

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
    kmeans = KMeans(n_clusters=num_segments - 1, random_state=0, n_init=10).fit(features)
    
    # Get segmentation points
    cluster_centers = kmeans.cluster_centers_
    segment_indices = cluster_centers[:, 0].astype(int)
    segment_indices.sort()  # Ensure segmentation points are ordered
    
    # Make sure segment points are within valid range
    segment_indices = np.clip(segment_indices, 1, num_frames - 2)
    
    return segment_indices.tolist()

def visualize_segmentation(skeleton_sequence, segment_points, optimal_segments=None):
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
    
    title = 'Skeleton Sequence Segmentation'
    if optimal_segments is not None:
        title += f' (Optimal: {optimal_segments} segments)'
    
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Average Joint Position')
    plt.show()

def representative_segs(skeleton_batch, segment_points_batch):
    # Ensure input is PyTorch tensor
    if not isinstance(skeleton_batch, torch.Tensor):
        skeleton_batch = torch.tensor(skeleton_batch)
    
    batch_size, embedding_size, num_frames = skeleton_batch.shape
    
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
                segment_frames = skeleton_sequence[:, start_frame]
            else:
                segment_frames = skeleton_sequence[:, start_frame:end_frame]
                rep_seg = torch.mean(segment_frames, dim=1)     # torch.Size([256, 25])
            
            # Add to collection
            all_rep_segs.append(rep_seg)
    
    # Stack all representative segments into a single tensor
    # Shape: [total_segments, embedding_size, joints]
    if all_rep_segs:
        rep_segs = torch.stack(all_rep_segs, dim=0)     # torch.Size([80, 256, 25])
    else:
        # Handle empty case
        rep_segs = torch.zeros(0, embedding_size)
    
    return rep_segs


if __name__ == "__main__":
    dataset_path = "/home/penghan/HAR-MLDA/STAR/fl_features/sample"
    batch_size = 4
    ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
    train_loader = ntu_loaders.get_train_loader(batch_size, 0)
    
    for i, (inputs, target) in enumerate(train_loader):
        skeleton_batch = inputs
        
        # Example 1: Original functionality with fixed number of segments
        print("=== Original functionality ===")
        segment_points_batch = segment_skeleton_sequences_with_dtw(skeleton_batch, num_segments=[3, 4, 2, 3])
        
        for i, points in enumerate(segment_points_batch):
            print(f"Sample {i} segment points: {points}")
        
        # Example 2: New adaptive functionality with range
        print("\n=== Adaptive segmentation ===")
        segment_points_batch_adaptive, optimal_segments_batch = segment_skeleton_sequences_with_dtw(
            skeleton_batch, range_seg=[2, 4]
        )
        
        for i, (points, optimal_segs) in enumerate(zip(segment_points_batch_adaptive, optimal_segments_batch)):
            print(f"Sample {i} optimal segments: {optimal_segs}, segment points: {points}")
        
        break 