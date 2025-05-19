import numpy as np
import torch
import torch.nn.functional as F
from data_cnn60_origin import AverageMeter, NTUDataLoaders
import ipdb
def smooth_min(x, gamma=0.1):
    """
    Implements the smoothMin operator from the paper.
    
    Args:
        x: Input tensor of shape (batch_size, n)
        gamma: Temperature parameter
    
    Returns:
        Smooth approximation of the minimum
    """
    if gamma == 0:
        return torch.min(x, dim=-1, keepdim=True)[0]
    
    exp_neg_x = torch.exp(-x / gamma)
    sum_exp = torch.sum(exp_neg_x, dim=-1, keepdim=True)
    weights = exp_neg_x / sum_exp
    return torch.sum(x * weights, dim=-1, keepdim=True)


def pair_cost(xi, yj, beta=0.1):
    """
    Computes the contrastive cost for matching xi to yj.
    
    Args:
        xi: Embedding vector of shape (batch_size, embedding_dim)
        yj: All embedding vectors of shape (batch_size, n, embedding_dim)
        beta: Temperature parameter
    
    Returns:
        Cost of matching xi to yj
    """
    # L2 normalize
    xi_norm = F.normalize(xi, p=2, dim=-1)
    yj_norm = F.normalize(yj, p=2, dim=-1)
    
    # Compute similarity scores
    sim = torch.matmul(xi_norm.unsqueeze(1), yj_norm.transpose(-2, -1)).squeeze(1)
    
    # Apply softmax along the second dimension (columns)
    log_probs = F.log_softmax(sim / beta, dim=-1)
    
    # Extract the cost for the specific yj
    # Since we want log P(j|i), we extract the corresponding column
    # Note: In practice, you would need to index properly to get the jth element
    return -log_probs  # Shape: (batch_size, n)


def smooth_dtw(X, Y, gamma=0.1, beta=0.1):
    """
    Implements the smoothDTW algorithm from the paper.
    
    Args:
        X: First sequence of shape (batch_size, m, embedding_dim)
        Y: Second sequence of shape (batch_size, n, embedding_dim)
        gamma: Temperature parameter for smoothMin
        beta: Temperature parameter for cost function
    
    Returns:
        Accumulated cost matrix R
    """
    batch_size, m, d = X.shape
    _, n, _ = Y.shape
    
    # Initialize accumulated cost matrix R
    R = torch.zeros((batch_size, m+1, n+1), device=X.device)
    R[:, 0, 1:] = float('inf')
    R[:, 1:, 0] = float('inf')
    
    # Compute pairwise cost matrix
    C = torch.zeros((batch_size, m, n), device=X.device)
    for i in range(m):
        for j in range(n):
            C[:, i, j] = pair_cost(X[:, i], Y[:, j], beta=beta)[:, j]
    
    # Fill the accumulated cost matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            r_ij = torch.stack([
                R[:, i-1, j-1], 
                R[:, i-1, j], 
                R[:, i, j-1]
            ], dim=-1)
            
            R[:, i, j] = C[:, i-1, j-1] + smooth_min(r_ij, gamma=gamma).squeeze(-1)
    
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
    frames, embedding_size = embeddings.shape
    
    # Convert to PyTorch tensor if it's not already
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    # Add batch dimension
    embeddings = embeddings.unsqueeze(0)  # (1, frames, embedding_size)
    
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
        ipdb.set_trace()
        s = inputs
        s = s.mean(dim=3)   # 4, 256, 16
        s = s.permute(0,2,1) # 4, 16, 256
        for _, embeddings in enumerate(s):
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