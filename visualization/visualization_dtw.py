def visualize_cost_matrix(X, Y=None, beta=0.1, figsize=(10, 8), cmap='viridis', title=None):
    """
    Visualizes the pairwise cost matrix between two sequences of embeddings.
    If Y is None, computes the self-similarity matrix of X.
    
    Args:
        X: First sequence of embeddings, shape (frames_X, embedding_size)
        Y: Second sequence of embeddings, shape (frames_Y, embedding_size), optional
        beta: Temperature parameter for cost function
        figsize: Figure size for the plot
        cmap: Colormap for the visualization
        title: Optional title for the plot
    
    Returns:
        fig, ax: The matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    # Convert inputs to PyTorch tensors if they're not already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
    # If Y is not provided, use X for self-similarity
    if Y is None:
        Y = X
        self_similarity = True
    else:
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        self_similarity = False
    
    # Add batch dimension if not present
    if X.dim() == 2:
        X = X.unsqueeze(0)  # (1, frames_X, embedding_size)
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)  # (1, frames_Y, embedding_size)
    
    # L2 normalize the embeddings
    X_norm = F.normalize(X, p=2, dim=-1)
    Y_norm = F.normalize(Y, p=2, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(X_norm, Y_norm.transpose(-2, -1))  # (1, frames_X, frames_Y)
    
    # Convert to cost matrix (negative similarity / beta)
    cost_matrix = -similarity / beta
    
    # Convert to numpy for visualization
    cost_matrix = cost_matrix.squeeze(0).cpu().numpy()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the cost matrix
    im = ax.imshow(cost_matrix, cmap=cmap)
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Cost')
    
    # Set labels and title
    ax.set_xlabel('Y frames' if not self_similarity else 'Frames')
    ax.set_ylabel('X frames' if not self_similarity else 'Frames')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Self-Similarity Matrix' if self_similarity else 'Pairwise Cost Matrix')
    
    # Add grid
    ax.set_xticks(np.arange(-.5, cost_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, cost_matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add frame indices
    ax.set_xticks(np.arange(cost_matrix.shape[1]))
    ax.set_yticks(np.arange(cost_matrix.shape[0]))
    ax.set_xticklabels(np.arange(cost_matrix.shape[1]))
    ax.set_yticklabels(np.arange(cost_matrix.shape[0]))
    
    plt.tight_layout()
    
    return fig, ax


def visualize_dtw_alignment(X, Y=None, gamma=0.1, beta=0.1, figsize=(15, 10), cmap='viridis', 
                           title=None, show_path=True, show_cost=True):
    """
    Visualizes the DTW alignment between two sequences, including cost matrix,
    accumulated cost matrix, and optimal path.
    
    Args:
        X: First sequence of embeddings, shape (frames_X, embedding_size)
        Y: Second sequence of embeddings, shape (frames_Y, embedding_size), optional
        gamma: Temperature parameter for smoothMin
        beta: Temperature parameter for cost function
        figsize: Figure size for the plot
        cmap: Colormap for the visualization
        title: Optional title for the plot
        show_path: Whether to show the optimal path
        show_cost: Whether to show the cost matrix alongside the accumulated cost
    
    Returns:
        fig, axs: The matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    # Convert inputs to PyTorch tensors if they're not already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
    # If Y is not provided, use X for self-similarity
    if Y is None:
        Y = X
        self_similarity = True
    else:
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float32)
        self_similarity = False
    
    # Add batch dimension if not present
    if X.dim() == 2:
        X = X.unsqueeze(0)  # (1, frames_X, embedding_size)
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)  # (1, frames_Y, embedding_size)
    
    batch_size, m, d = X.shape
    _, n, _ = Y.shape
    
    # Compute pairwise cost matrix
    X_norm = F.normalize(X, p=2, dim=-1)
    Y_norm = F.normalize(Y, p=2, dim=-1)
    similarity = torch.matmul(X_norm, Y_norm.transpose(-2, -1))
    cost_matrix = -similarity / beta
    
    # Compute smooth DTW
    R = torch.zeros((batch_size, m+1, n+1), device=X.device)
    R[:, 0, 1:] = float('inf')
    R[:, 1:, 0] = float('inf')
    
    # Fill the accumulated cost matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            prev_costs = torch.stack([
                R[:, i-1, j-1], 
                R[:, i-1, j], 
                R[:, i, j-1]
            ], dim=-1)
            
            # Apply smooth min
            if gamma == 0:
                min_prev = torch.min(prev_costs, dim=-1)[0]
            else:
                exp_neg_prev = torch.exp(-prev_costs / gamma)
                sum_exp = torch.sum(exp_neg_prev, dim=-1, keepdim=True)
                weights = exp_neg_prev / sum_exp
                min_prev = torch.sum(prev_costs * weights, dim=-1)
            
            R[:, i, j] = cost_matrix[:, i-1, j-1] + min_prev
    
    # Extract matrices for visualization
    cost_matrix_np = cost_matrix.squeeze(0).cpu().numpy()
    accumulated_cost_np = R.squeeze(0).cpu().numpy()[1:, 1:]  # Remove padding
    
    # Find the optimal path through backtracking
    if show_path:
        path = []
        i, j = m, n
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # Find which direction to move
            candidates = [
                R[0, i-1, j-1].item(),  # Diagonal
                R[0, i-1, j].item(),    # Up
                R[0, i, j-1].item()     # Left
            ]
            
            min_idx = np.argmin(candidates)
            
            if min_idx == 0:  # Diagonal
                i -= 1
                j -= 1
            elif min_idx == 1:  # Up
                i -= 1
            else:  # Left
                j -= 1
        
        path.reverse()  # To get path from (0,0) to (m-1,n-1)
    
    # Create the figure
    if show_cost:
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axs
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=figsize)
        axs = [ax2]
    
    # Plot the cost matrix
    if show_cost:
        im1 = ax1.imshow(cost_matrix_np, cmap=cmap)
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label('Cost')
        ax1.set_title('Pairwise Cost Matrix')
        ax1.set_xlabel('Y frames' if not self_similarity else 'Frames')
        ax1.set_ylabel('X frames' if not self_similarity else 'Frames')
        
        # Add grid
        ax1.set_xticks(np.arange(-.5, cost_matrix_np.shape[1], 1), minor=True)
        ax1.set_yticks(np.arange(-.5, cost_matrix_np.shape[0], 1), minor=True)
        ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add frame indices
        ax1.set_xticks(np.arange(cost_matrix_np.shape[1]))
        ax1.set_yticks(np.arange(cost_matrix_np.shape[0]))
        ax1.set_xticklabels(np.arange(cost_matrix_np.shape[1]))
        ax1.set_yticklabels(np.arange(cost_matrix_np.shape[0]))
    
    # Plot the accumulated cost matrix
    im2 = ax2.imshow(accumulated_cost_np, cmap=cmap)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Accumulated Cost')
    
    if title:
        ax2.set_title(title)
    else:
        ax2.set_title('Accumulated Cost Matrix with DTW Path')
    
    ax2.set_xlabel('Y frames' if not self_similarity else 'Frames')
    ax2.set_ylabel('X frames' if not self_similarity else 'Frames')
    
    # Add grid
    ax2.set_xticks(np.arange(-.5, accumulated_cost_np.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-.5, accumulated_cost_np.shape[0], 1), minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add frame indices
    ax2.set_xticks(np.arange(accumulated_cost_np.shape[1]))
    ax2.set_yticks(np.arange(accumulated_cost_np.shape[0]))
    ax2.set_xticklabels(np.arange(accumulated_cost_np.shape[1]))
    ax2.set_yticklabels(np.arange(accumulated_cost_np.shape[0]))
    
    # Plot the optimal path
    if show_path:
        path_x, path_y = zip(*path)
        ax2.plot(path_y, path_x, 'r-', linewidth=2)
        ax2.scatter(path_y, path_x, c='r', s=50)
        
        if show_cost:
            ax1.plot(path_y, path_x, 'r-', linewidth=2)
            ax1.scatter(path_y, path_x, c='r', s=50)
    
    plt.tight_layout()
    
    return fig, axs


def visualize_segments_with_cost(embeddings, num_segments, gamma=0.1, beta=0.1, 
                              figsize=(18, 10), cmap='viridis'):
    """
    Visualizes the temporal segmentation along with the DTW cost matrices.
    
    Args:
        embeddings: Sequence of embeddings, shape (frames, embedding_size)
        num_segments: Number of segments to divide the sequence into
        gamma: Temperature parameter for smoothMin
        beta: Temperature parameter for cost function
        figsize: Figure size for the plot
        cmap: Colormap for the visualization
    
    Returns:
        segment_indices: Indices of the segment boundaries
        segmented_embeddings: List of embeddings for each segment
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    import torch
    
    # Convert to PyTorch tensor if it's not already
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    # Perform temporal segmentation
    segment_indices, segmented_embeddings = temporal_segment(
        embeddings, num_segments, gamma, beta
    )
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1]})
    
    # Subplot 1: Self-similarity cost matrix
    ax1 = axs[0]
    X = embeddings.unsqueeze(0) if embeddings.dim() == 2 else embeddings
    X_norm = torch.nn.functional.normalize(X, p=2, dim=-1)
    similarity = torch.matmul(X_norm, X_norm.transpose(-2, -1))
    cost_matrix = -similarity / beta
    cost_matrix_np = cost_matrix.squeeze(0).cpu().numpy()
    
    im1 = ax1.imshow(cost_matrix_np, cmap=cmap)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Self-Similarity Cost Matrix')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Frames')
    
    # Add grid
    ax1.set_xticks(np.arange(-.5, cost_matrix_np.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-.5, cost_matrix_np.shape[0], 1), minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Mark segment boundaries
    for idx in segment_indices:
        if idx > 0 and idx < embeddings.shape[0]:
            ax1.axhline(y=idx-0.5, color='r', linestyle='-', linewidth=2)
            ax1.axvline(x=idx-0.5, color='r', linestyle='-', linewidth=2)
    
    # Subplot 2: PCA visualization of embeddings
    ax2 = axs[1]
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Plot embeddings with segment colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    
    for i in range(len(segment_indices) - 1):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        color = colors[i % len(colors)]
        
        ax2.scatter(
            embeddings_2d[start:end, 0],
            embeddings_2d[start:end, 1],
            c=color,
            label=f'Segment {i+1}'
        )
    
    # Connect points to show sequence
    ax2.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'k-', alpha=0.3)
    
    # Mark segment boundaries
    for idx in segment_indices:
        if idx > 0 and idx < len(embeddings_np):
            ax2.scatter(
                embeddings_2d[idx-1, 0],
                embeddings_2d[idx-1, 1],
                c='black',
                marker='x',
                s=100
            )
    
    ax2.set_title('Embedding Visualization (PCA)')
    ax2.legend()
    
    # Subplot 3: Segment information and heatmap
    ax3 = axs[2]
    
    # Create a segment heatmap
    segment_heatmap = np.zeros((embeddings_np.shape[0], num_segments))
    
    for i in range(num_segments):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        segment_heatmap[start:end, i] = 1
    
    im3 = ax3.imshow(segment_heatmap, cmap='Blues', aspect='auto')
    fig.colorbar(im3, ax=ax3)
    
    ax3.set_title('Segment Assignments')
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Frame Index')
    
    # Add segment boundary markers
    for idx in segment_indices:
        if idx > 0 and idx < embeddings_np.shape[0]:
            ax3.axhline(y=idx-0.5, color='r', linestyle='-', linewidth=2)
    
    # Add segment lengths as text
    for i in range(num_segments):
        start = segment_indices[i]
        end = segment_indices[i + 1]
        length = end - start
        center_y = (start + end) / 2
        ax3.text(i, center_y, f"{length}", ha='center', va='center', color='black', fontweight='bold')
    
    # Set ticks
    ax3.set_xticks(np.arange(num_segments))
    ax3.set_xticklabels([f"S{i+1}" for i in range(num_segments)])
    
    plt.tight_layout()
    plt.show()
    
    return segment_indices, segmented_embeddings


def demo_visualizations():
    """
    Demonstrates the visualization functions with sample data.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    # Generate sample data
    frames = 16
    embedding_size = 256
    num_segments = 4
    
    # Create structured random embeddings
    np.random.seed(42)
    
    # Base embeddings for each segment
    base_embeddings = np.random.randn(num_segments, embedding_size)
    
    # Create frames with structure
    embeddings = np.zeros((frames, embedding_size))
    for i in range(frames):
        # Determine which base embedding to use
        segment = int(i * num_segments / frames)
        # Add noise
        noise = np.random.randn(embedding_size) * 0.2
        embeddings[i] = base_embeddings[segment] + noise
    
    # 1. Visualize pairwise cost matrix
    print("Visualizing pairwise cost matrix...")
    fig1, ax1 = visualize_cost_matrix(embeddings, beta=0.1)
    plt.tight_layout()
    plt.show()
    
    # 2. Visualize DTW alignment
    print("Visualizing DTW alignment...")
    # Create a distorted copy of the sequence for alignment demonstration
    distorted = np.zeros((frames, embedding_size))
    
    # Add non-linear distortion to the timing
    for i in range(frames):
        # Apply non-linear distortion to indices
        original_idx = min(frames - 1, int(i ** 1.2))
        distorted[i] = embeddings[original_idx] + np.random.randn(embedding_size) * 0.1
    
    fig2, axs2 = visualize_dtw_alignment(embeddings, distorted, gamma=0.1, beta=0.1)
    plt.tight_layout()
    plt.show()
    
    # 3. Visualize segmentation with cost matrices
    print("Visualizing segmentation with cost matrices...")
    segment_indices, _ = visualize_segments_with_cost(embeddings, num_segments, gamma=0.1, beta=0.1)
    
    print(f"Segment boundaries: {segment_indices}")
    for i in range(len(segment_indices) - 1):
        print(f"Segment {i+1}: frames {segment_indices[i]} to {segment_indices[i+1] - 1}")


if __name__ == "__main__":
    demo_visualizations()