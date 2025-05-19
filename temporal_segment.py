import numpy as np
import torch
import torch.nn.functional as F
from data_cnn60_origin import AverageMeter, NTUDataLoaders
import ipdb

def smooth_min(a, gamma=0.1):
    """
    Stable implementation of smoothMin operation.
    
    Args:
        a: Input tensor
        gamma: Temperature parameter
    
    Returns:
        Smooth approximation of the minimum
    """
    if gamma <= 0:
        return torch.min(a, dim=-1)[0]
    
    # Subtract the minimum value for numerical stability
    a_min = torch.min(a, dim=-1, keepdim=True)[0]
    exp_term = torch.exp(-(a - a_min) / gamma)
    sum_exp = torch.sum(exp_term, dim=-1, keepdim=True)
    weights = exp_term / (sum_exp + 1e-10)  # Add epsilon to avoid division by zero
    
    return torch.sum(a * weights, dim=-1)


def cosine_similarity(x, y):
    """
    Compute cosine similarity between two batches of vectors.
    
    Args:
        x: First batch of vectors, shape (batch_size, embedding_dim)
        y: Second batch of vectors, shape (batch_size, embedding_dim)
        
    Returns:
        Cosine similarity, shape (batch_size,)
    """
    # Normalize
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    
    # Compute similarity
    sim = torch.sum(x_norm * y_norm, dim=-1)
    
    # Ensure in range [-1, 1] for numerical stability
    sim = torch.clamp(sim, -1.0, 1.0)
    
    return sim


def smooth_dtw(X, Y, gamma=0.1, beta=0.1):
    """
    Numerically stable implementation of smoothDTW.
    
    Args:
        X: First sequence, shape (batch_size, m, embedding_dim)
        Y: Second sequence, shape (batch_size, n, embedding_dim)
        gamma: Temperature parameter for smoothMin
        beta: Temperature parameter for similarity scaling
        
    Returns:
        Accumulated cost matrix
    """
    batch_size, m, embedding_dim = X.shape
    _, n, _ = Y.shape
    
    # Initialize the accumulated cost matrix with zeros and infinity
    R = torch.full((batch_size, m+1, n+1), float('inf'), device=X.device, dtype=X.dtype)
    R[:, 0, 0] = 0.0
    
    # Pre-compute all pairwise costs for efficiency and stability
    cost_matrix = torch.zeros((batch_size, m, n), device=X.device, dtype=X.dtype)
    
    for i in range(m):
        for j in range(n):
            # Compute similarity score (higher = better match)
            sim = cosine_similarity(X[:, i], Y[:, j])
            cost = sim / beta
            cost_matrix[:, i, j] = cost
    
    # Fill the accumulated cost matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            prev_costs = torch.stack([
                R[:, i-1, j-1],  # Diagonal
                R[:, i-1, j],    # Up
                R[:, i, j-1]     # Left
            ], dim=-1)
            
            # Apply smooth min
            smooth_min_val = smooth_min(prev_costs, gamma)
            
            # Add current cost
            R[:, i, j] = cost_matrix[:, i-1, j-1] + smooth_min_val
    
    return R


def temporal_segment(embeddings, num_segments, gamma=0.1, beta=0.1):
    """
    Segments a sequence of embeddings into the specified number of segments.
    
    Args:
        embeddings: Sequence of embeddings of shape (frames, embedding_size)
        num_segments: Number of segments to divide the sequence into
        gamma: Temperature parameter for smoothMin
        beta: Temperature parameter for cost function
    
    Returns:
        segment_indices: Indices of the frames at the segment boundaries
        segmented_embeddings: List of embeddings for each segment
    """
    frames, embedding_size = embeddings.shape
    
    # Convert to PyTorch tensor if it's not already
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    # Add batch dimension
    embeddings = embeddings.unsqueeze(0)  # (1, frames, embedding_size)
    
    # Create an idealized uniform segmentation as a reference
    ideal_length = frames / num_segments
    ideal_positions = [int(i * ideal_length) for i in range(num_segments + 1)]
    ideal_segments = []
    
    # Create the idealized segments
    for i in range(num_segments):
        start = ideal_positions[i]
        end = ideal_positions[i + 1]
        if start != end:  # Avoid empty segments
            segment = embeddings[:, start:end, :]
            ideal_segments.append(segment)
    
    # Compute the alignment between the sequence and the idealized segmentation
    # For simplicity, we'll use uniform spacing as an initialization
    segment_indices = [0]
    
    # Align the sequence with the ideal segments using DTW
    for i in range(num_segments - 1):
        remaining_frames = frames - segment_indices[-1]
        remaining_segments = num_segments - i
        ideal_size = remaining_frames // remaining_segments
        
        # Find optimal cut point
        best_cut = segment_indices[-1] + ideal_size
        if best_cut < frames:
            segment_indices.append(best_cut)
    
    segment_indices.append(frames)  # Add the last index
    
    # Refine the segmentation using DTW
    max_iterations = 5
    for _ in range(max_iterations):
        updated = False
        
        for i in range(1, num_segments):
            # Try to adjust each segment boundary to improve overall alignment
            current_idx = segment_indices[i]
            
            # Check a window around the current boundary
            window_size = max(2, frames // (num_segments * 2))
            best_idx = current_idx
            best_cost = float('inf')
            
            for j in range(max(segment_indices[i-1]+1, current_idx-window_size), 
                          min(segment_indices[i+1], current_idx+window_size+1)):
                # Create temporary segmentation with this boundary
                temp_indices = segment_indices.copy()
                temp_indices[i] = j
                
                # Evaluate segmentation cost
                cost = 0
                for k in range(num_segments):
                    segment = embeddings[:, temp_indices[k]:temp_indices[k+1], :]
                    if segment.shape[1] > 1:  # Skip empty segments
                        # Self-similarity within segment should be high
                        sim_matrix = torch.matmul(
                            F.normalize(segment, p=2, dim=-1),
                            F.normalize(segment, p=2, dim=-1).transpose(-2, -1)
                        )
                        # We want high similarity (low cost)
                        cost -= torch.mean(sim_matrix)
                
                if cost < best_cost:
                    best_cost = cost
                    best_idx = j
            
            if best_idx != current_idx:
                segment_indices[i] = best_idx
                updated = True
        
        if not updated:
            break
    
    # Create the final segmented embeddings
    segmented_embeddings = []
    for i in range(num_segments):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        segment = embeddings[0, start:end, :].detach().cpu().numpy()
        segmented_embeddings.append(segment)
    
    return segment_indices, segmented_embeddings


def temporal_segment_advanced(embeddings, num_segments, gamma=0.1, beta=0.1):
    """
    An improved version of temporal segmentation using principles from the paper.
    This version uses DTW to align the sequence with an idealized uniformly segmented sequence.
    
    Args:
        embeddings: Sequence of embeddings of shape (frames, embedding_size)
        num_segments: Number of segments to divide the sequence into
        gamma: Temperature parameter for smoothMin
        beta: Temperature parameter for cost function
    
    Returns:
        segment_indices: Indices of the frames at the segment boundaries
        segmented_embeddings: List of embeddings for each segment
    """
    bs, frames, embedding_size = embeddings.shape       # 16,256
    
    # Convert to PyTorch tensor if it's not already
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
    # Create uniform reference points to represent segment boundaries
    segment_length = frames / num_segments
    reference_points = []
    
    # Generate reference embeddings for each segment
    for i in range(num_segments):
        # Create a representative embedding for each segment
        start = int(i * segment_length)
        end = int((i + 1) * segment_length)
        if end > frames:
            end = frames
        
        # Use the mean embedding of frames in the ideal segment as reference
        if start < end:
            segment_mean = torch.mean(embeddings[:, start:end, :], dim=1)
            reference_points.append(segment_mean)
    
    reference_points = torch.stack(reference_points, dim=1)  # (1, num_segments, embedding_size)
    ipdb.set_trace()
    # Compute the optimal alignment between the sequence and the reference points
    R = smooth_dtw(embeddings, reference_points, gamma=gamma, beta=beta)
    
    # Backtrack to find the optimal path
    i, j = frames, num_segments
    path = [(i-1, j-1)]
    
    while i > 1 and j > 1:
        candidates = [
            (i-1, j-1),
            (i-1, j),
            (i, j-1)
        ]
        
        min_idx = 0
        min_cost = float('inf')
        
        for idx, (ci, cj) in enumerate(candidates):
            if ci >= 0 and cj >= 0 and R[0, ci, cj] < min_cost:
                min_cost = R[0, ci, cj]
                min_idx = idx
        
        next_i, next_j = candidates[min_idx]
        
        if next_j != j:  # We've moved to a new segment
            path.append((next_i, next_j))
        
        i, j = next_i, next_j
    
    # Extract segment boundaries from the path
    segment_indices = [0]
    current_segment = 0
    
    for frame_idx, segment_idx in sorted(path):
        if segment_idx > current_segment:
            segment_indices.append(frame_idx + 1)
            current_segment = segment_idx
    
    if segment_indices[-1] != frames:
        segment_indices.append(frames)
    
    # Ensure we have the correct number of segments
    while len(segment_indices) < num_segments + 1:
        # Find the largest gap and add a boundary in the middle
        max_gap = 0
        max_gap_idx = 0
        
        for i in range(len(segment_indices) - 1):
            gap = segment_indices[i+1] - segment_indices[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
        
        new_boundary = segment_indices[max_gap_idx] + max_gap // 2
        segment_indices.insert(max_gap_idx + 1, new_boundary)
    
    # If we have too many segments, merge the smallest ones
    while len(segment_indices) > num_segments + 1:
        min_gap = float('inf')
        min_gap_idx = 0
        
        for i in range(len(segment_indices) - 1):
            gap = segment_indices[i+1] - segment_indices[i]
            if gap < min_gap:
                min_gap = gap
                min_gap_idx = i
        
        segment_indices.pop(min_gap_idx + 1)
    
    # Sort the indices to ensure they're in ascending order
    segment_indices.sort()
    
    # Create the final segmented embeddings
    segmented_embeddings = []
    for i in range(num_segments):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        segment = embeddings[0, start:end, :].detach().cpu().numpy()
        segmented_embeddings.append(segment)
    
    return segment_indices, segmented_embeddings

def demo_temporal_segment():
    """
    Demonstrates the temporal segmentation algorithm on random data.
    """
    import matplotlib.pyplot as plt
    dataset_path = "/home/penghan/HAR-MLDA/STAR/fl_features/shift_ntu60_5_r"
    batch_size = 4
    ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
    train_loader = ntu_loaders.get_train_loader(batch_size, 0)
    # Create a random sequence
    frames = 16
    embedding_size = 256
    np.random.seed(42)  # For reproducibility
    
    for i, (inputs, target) in enumerate(train_loader):
        s = inputs
        s = s.mean(dim=3)   # 4, 256, 16
        s = s.permute(0,2,1) # 4, 16, 256
        embeddings = s
        num_segments = 3
        segment_indices, segmented_embeddings = temporal_segment_advanced(
            embeddings, num_segments, gamma=0.1, beta=0.1
        )
        
        print(f"Segment boundaries: {segment_indices}")
        for i, segment in enumerate(segmented_embeddings):
            print(f"Segment {i}: {segment.shape[0]} frames")
        
        # Visualize the segmentation
        # For visualization purposes, we'll use PCA to reduce the dimensionality
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 6))
        
        # Plot the original sequence
        plt.subplot(1, 2, 1)
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(frames))
        plt.colorbar(label='Frame index')
        plt.title('Original sequence')
        
        # Plot the segmented sequence
        plt.subplot(1, 2, 2)
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        for i in range(num_segments):
            start = segment_indices[i]
            end = segment_indices[i + 1]
            plt.scatter(
                embeddings_2d[start:end, 0], 
                embeddings_2d[start:end, 1], 
                c=colors[i % len(colors)],
                label=f'Segment {i}'
            )
        
        plt.legend()
        plt.title('Segmented sequence')
        plt.tight_layout()
        plt.show()
    
    return segment_indices, segmented_embeddings

# Run the demo
if __name__ == "__main__":
    segment_indices, segmented_embeddings = demo_temporal_segment()