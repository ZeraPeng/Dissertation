import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import logging


class TemporalAlignment:
    """
    Class for temporal alignment of skeleton sequences 
    encoded by VAE to separate semantic-related parts.
    """
    def __init__(self, method='dtw', n_segments=None, similarity_threshold=0.7):
        """
        Initialize the temporal alignment module.
        
        Args:
            method (str): Method for temporal alignment. Options: 'dtw', 'clustering', 'hmm', 'spectral'
            n_segments (int, optional): Number of segments to divide the sequence into
            similarity_threshold (float): Threshold for similarity when grouping embeddings
        """
        self.method = method
        self.n_segments = n_segments
        self.similarity_threshold = similarity_threshold
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('TemporalAlignment')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def align(self, embeddings, reference=None):
        """
        Align temporal embeddings using the selected method.
        
        Args:
            embeddings (torch.Tensor): Sequence of embeddings [seq_len, embed_dim]
            reference (torch.Tensor, optional): Reference sequence for alignment
            
        Returns:
            list: List of segment indices [(start1, end1), (start2, end2), ...]
            list: List of segment embeddings [embed1, embed2, ...]
        """
        if self.method == 'dtw':
            return self._align_dtw(embeddings, reference)
        elif self.method == 'clustering':
            return self._align_clustering(embeddings)
        elif self.method == 'hmm':
            return self._align_hmm(embeddings)
        elif self.method == 'spectral':
            return self._align_spectral(embeddings)
        elif self.method == 'sliding_window':
            return self._align_sliding_window(embeddings)
        else:
            raise ValueError(f"Unknown alignment method: {self.method}")
    
    def _align_dtw(self, embeddings, reference=None):
        """
        Align using Dynamic Time Warping.
        If reference is None, finds changepoints within the sequence.
        """
        self.logger.info("Aligning with DTW...")
        
        # Convert to numpy if it's a torch tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        if reference is not None:
            if isinstance(reference, torch.Tensor):
                reference = reference.detach().cpu().numpy()
            
            # Compute DTW distance
            distance, path = fastdtw(embeddings, reference, dist=euclidean)
            
            # Use the warping path to create aligned segments
            aligned_path = np.array(path)
            segments = []
            current_ref_idx = aligned_path[0, 1]
            start_idx = aligned_path[0, 0]
            
            for i in range(1, len(aligned_path)):
                if aligned_path[i, 1] != current_ref_idx:
                    segments.append((start_idx, aligned_path[i-1, 0]))
                    start_idx = aligned_path[i, 0]
                    current_ref_idx = aligned_path[i, 1]
            
            # Add the last segment
            segments.append((start_idx, aligned_path[-1, 0]))
            
        else:
            # Use DTW as a distance metric to find changepoints
            # Compute distance matrix
            n = len(embeddings)
            dist_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    dist, _ = fastdtw(embeddings[i:i+1], embeddings[j:j+1], dist=euclidean)
                    dist_matrix[i, j] = dist_matrix[j, i] = dist
            
            # Use distance matrix with DBSCAN for segmentation
            clustering = DBSCAN(eps=self.similarity_threshold, min_samples=3, metric='precomputed')
            labels = clustering.fit_predict(dist_matrix)
            
            # Create segments from labels
            segments = []
            current_label = labels[0]
            start_idx = 0
            
            for i in range(1, len(labels)):
                if labels[i] != current_label or labels[i] == -1:  # New segment or noise
                    if current_label != -1:  # Ignore noise segments
                        segments.append((start_idx, i-1))
                    start_idx = i
                    current_label = labels[i]
            
            # Add the last segment if it's not noise
            if current_label != -1:
                segments.append((start_idx, len(labels)-1))
                
        # Compute segment embeddings (average of embeddings in each segment)
        segment_embeddings = []
        for start, end in segments:
            if isinstance(embeddings, np.ndarray):
                segment_embeddings.append(np.mean(embeddings[start:end+1], axis=0))
            else:
                segment_embeddings.append(torch.mean(embeddings[start:end+1], dim=0))
        
        return segments, segment_embeddings
    
    def _align_clustering(self, embeddings):
        """
        Align by clustering similar embeddings together while preserving temporal order.
        """
        self.logger.info("Aligning with clustering...")
        
        # Convert to numpy if it's a torch tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Determine number of clusters if not specified
        n_clusters = self.n_segments if self.n_segments is not None else min(len(embeddings) // 5, 10)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        
        # Find temporal segments with same labels
        segments = []
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append((start_idx, i-1))
                start_idx = i
                current_label = labels[i]
        
        # Add the last segment
        segments.append((start_idx, len(labels)-1))
        
        # Compute segment embeddings
        segment_embeddings = []
        for start, end in segments:
            segment_embeddings.append(np.mean(embeddings[start:end+1], axis=0))
        
        return segments, segment_embeddings
    
    def _align_hmm(self, embeddings):
        """
        Align using Hidden Markov Models.
        Requires hmmlearn package.
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            self.logger.error("hmmlearn package is required for HMM alignment")
            raise ImportError("hmmlearn package is required")
        
        self.logger.info("Aligning with HMM...")
        
        # Convert to numpy if it's a torch tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Reshape for hmmlearn
        X = embeddings.reshape(-1, embeddings.shape[-1])
        
        # Determine number of states if not specified
        n_states = self.n_segments if self.n_segments is not None else min(len(embeddings) // 5, 10)
        
        # Train HMM
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=0)
        model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        # Find segments with same states
        segments = []
        current_state = states[0]
        start_idx = 0
        
        for i in range(1, len(states)):
            if states[i] != current_state:
                segments.append((start_idx, i-1))
                start_idx = i
                current_state = states[i]
        
        # Add the last segment
        segments.append((start_idx, len(states)-1))
        
        # Compute segment embeddings
        segment_embeddings = []
        for start, end in segments:
            segment_embeddings.append(np.mean(embeddings[start:end+1], axis=0))
        
        return segments, segment_embeddings
    
    def _align_spectral(self, embeddings):
        """
        Align using spectral clustering with temporal constraints.
        """
        from sklearn.cluster import SpectralClustering
        
        self.logger.info("Aligning with spectral clustering...")
        
        # Convert to numpy if it's a torch tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Compute similarity matrix
        sim_matrix = 1.0 - pairwise_distances(embeddings, metric='cosine')
        
        # Add temporal constraints - boost similarity of temporally adjacent frames
        n = len(embeddings)
        temporal_weight = 0.5
        for i in range(n):
            for j in range(n):
                time_diff = abs(i - j)
                if time_diff <= 3:  # Adjacent frames get boosted similarity
                    sim_matrix[i, j] += temporal_weight * (1.0 - time_diff / 3.0)
        
        # Clip values to [0, 1]
        sim_matrix = np.clip(sim_matrix, 0, 1)
        
        # Determine number of clusters if not specified
        n_clusters = self.n_segments if self.n_segments is not None else min(len(embeddings) // 5, 10)
        
        # Apply spectral clustering
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                        affinity='precomputed',
                                        random_state=0)
        labels = clustering.fit_predict(sim_matrix)
        
        # Find segments with same labels while enforcing temporal continuity
        segments = []
        current_label = labels[0]
        start_idx = 0
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append((start_idx, i-1))
                start_idx = i
                current_label = labels[i]
        
        # Add the last segment
        segments.append((start_idx, len(labels)-1))
        
        # Compute segment embeddings
        segment_embeddings = []
        for start, end in segments:
            segment_embeddings.append(np.mean(embeddings[start:end+1], axis=0))
        
        return segments, segment_embeddings
    
    def _align_sliding_window(self, embeddings, window_size=5):
        """
        Align using sliding window with adaptive threshold.
        """
        self.logger.info("Aligning with sliding window...")
        
        # Convert to numpy if it's a torch tensor
        is_torch = isinstance(embeddings, torch.Tensor)
        if is_torch:
            embeddings = embeddings.detach().cpu().numpy()
            
        n = len(embeddings)
        
        # Initialize segments
        segments = []
        start_idx = 0
        distances = []
        
        # Compute distances between consecutive windows
        for i in range(0, n - window_size):
            window1 = embeddings[i:i+window_size]
            window2 = embeddings[i+1:i+window_size+1]
            
            # Compute average distance between windows
            dist = np.mean(np.linalg.norm(window1 - window2, axis=1))
            distances.append(dist)
        
        if len(distances) == 0:
            # Sequence too short, treat as one segment
            segments.append((0, n-1))
            segment_embeddings = [np.mean(embeddings, axis=0)]
            return segments, segment_embeddings
            
        # Compute adaptive threshold
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 1.5 * std_dist
        
        # Find segment boundaries based on distance peaks
        for i in range(window_size, n - window_size):
            idx = i - window_size
            if idx < len(distances) and distances[idx] > threshold:
                segments.append((start_idx, i-1))
                start_idx = i
        
        # Add the last segment
        segments.append((start_idx, n-1))
        
        # Compute segment embeddings
        segment_embeddings = []
        for start, end in segments:
            segment_embeddings.append(np.mean(embeddings[start:end+1], axis=0))
            
        # Convert back to torch if input was torch
        if is_torch:
            segment_embeddings = [torch.tensor(emb, device=embeddings.device) for emb in segment_embeddings]
        
        return segments, segment_embeddings


class TemporalClassifier:
    """
    Classifier for temporally aligned segments using pretrained semantic classifier.
    """
    def __init__(self, classifier, sequence_encoder, device=None):
        """
        Initialize the temporal classifier.
        
        Args:
            classifier: Pretrained classifier model
            sequence_encoder: Encoder to get embeddings from sequences
            device: Torch device
        """
        self.classifier = classifier
        self.sequence_encoder = sequence_encoder
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger('TemporalClassifier')
        self.logger.setLevel(logging.INFO)
    
    def classify_segments(self, segments, segment_embeddings):
        """
        Classify each segment using the pretrained classifier.
        
        Args:
            segments: List of segment indices [(start1, end1), (start2, end2), ...]
            segment_embeddings: List of segment embeddings [embed1, embed2, ...]
            
        Returns:
            list: Predicted labels for each segment
            list: Confidence scores for each segment
        """
        self.logger.info(f"Classifying {len(segments)} segments...")
        
        labels = []
        confidences = []
        
        # Set models to eval mode
        self.classifier.eval()
        self.sequence_encoder.eval()
        
        with torch.no_grad():
            for embedding in segment_embeddings:
                # Convert to tensor if it's numpy
                if isinstance(embedding, np.ndarray):
                    embedding = torch.tensor(embedding, dtype=torch.float32).to(self.device)
                elif isinstance(embedding, torch.Tensor):
                    embedding = embedding.to(self.device)
                
                # Reshape if needed
                if embedding.dim() == 1:
                    embedding = embedding.unsqueeze(0)
                
                # Get classifier output
                outputs = self.classifier(embedding)
                
                # Get predicted label and confidence
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Some classifiers return (outputs, features)
                
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                labels.append(predicted.item())
                confidences.append(confidence.item())
        
        return labels, confidences
    
    def classify_sequence(self, sequence, alignment_method='dtw', n_segments=None):
        """
        Align and classify a full sequence.
        
        Args:
            sequence: Input sequence
            alignment_method: Method for temporal alignment
            n_segments: Number of segments to divide the sequence into
            
        Returns:
            list: Segment boundaries
            list: Predicted labels for each segment
            list: Confidence scores for each segment
        """
        self.logger.info(f"Processing sequence with {alignment_method} alignment...")
        
        # Set encoder to eval mode
        self.sequence_encoder.eval()
        
        # Convert to tensor if it's numpy
        if isinstance(sequence, np.ndarray):
            sequence = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        elif isinstance(sequence, torch.Tensor):
            sequence = sequence.to(self.device)
            
        # Get embeddings
        with torch.no_grad():
            # Ensure sequence has batch dimension
            if sequence.dim() == 2:
                sequence = sequence.unsqueeze(0)
                
            # Get embeddings from encoder
            embeddings, _ = self.sequence_encoder(sequence)
            
            # Remove batch dimension
            embeddings = embeddings.squeeze(0)
        
        # Create aligner
        aligner = TemporalAlignment(method=alignment_method, n_segments=n_segments)
        
        # Align sequence
        segments, segment_embeddings = aligner.align(embeddings)
        
        # Classify segments
        labels, confidences = self.classify_segments(segments, segment_embeddings)
        
        return segments, labels, confidences

    def process_batch(self, batch, alignment_method='dtw'):
        """
        Process a batch of sequences.
        
        Args:
            batch: Batch of sequences [batch_size, seq_len, feat_dim]
            alignment_method: Method for temporal alignment
            
        Returns:
            list: List of segment boundaries for each sequence
            list: List of predicted labels for each sequence 
            list: List of confidence scores for each sequence
        """
        all_segments = []
        all_labels = []
        all_confidences = []
        
        for i in range(batch.size(0)):
            sequence = batch[i]
            segments, labels, confidences = self.classify_sequence(
                sequence, alignment_method=alignment_method)
            
            all_segments.append(segments)
            all_labels.append(labels)
            all_confidences.append(confidences)
            
        return all_segments, all_labels, all_confidences


def fuse_predictions(predictions_list, weights=None):
    """
    Fuse predictions from multiple models.
    
    Args:
        predictions_list: List of prediction sets [pred_set1, pred_set2, ...]
        weights: Weights for each prediction set
        
    Returns:
        numpy.ndarray: Fused predictions
    """
    if weights is None:
        weights = np.ones(len(predictions_list)) / len(predictions_list)
    
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # Stack and weight predictions
    weighted_preds = [w * pred for w, pred in zip(weights, predictions_list)]
    
    # Sum weighted predictions
    fused_preds = np.sum(np.stack(weighted_preds), axis=0)
    
    return fused_preds


# Utility function to visualize temporal segments
def visualize_segments(sequence, segments, labels=None, confidences=None):
    """
    Visualize temporal segments on the original sequence.
    
    Args:
        sequence: Original sequence data
        segments: List of segment boundaries [(start1, end1), (start2, end2), ...]
        labels: Optional list of labels for each segment
        confidences: Optional list of confidence scores
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot sequence (first few dimensions)
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy()
    
    n_dims = min(5, sequence.shape[1]) if len(sequence.shape) > 1 else 1
    for d in range(n_dims):
        if len(sequence.shape) > 1:
            ax.plot(sequence[:, d], alpha=0.5, label=f'Dim {d}')
        else:
            ax.plot(sequence, alpha=0.5, label='Sequence')
    
    # Add segment boundaries
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for i, ((start, end), color) in enumerate(zip(segments, colors)):
        label_text = f"Segment {i}"
        if labels is not None:
            label_text += f": {labels[i]}"
        if confidences is not None:
            label_text += f" ({confidences[i]:.2f})"
            
        ax.axvline(x=start, color=color, linestyle='--', alpha=0.7)
        ax.axvline(x=end, color=color, linestyle='--', alpha=0.7)
        
        # Add rectangle for segment
        rect = Rectangle((start, ax.get_ylim()[0]), end-start, 
                         ax.get_ylim()[1]-ax.get_ylim()[0], 
                         color=color, alpha=0.2)
        ax.add_patch(rect)
        
        # Add label
        ax.text((start+end)/2, ax.get_ylim()[1]*0.9, label_text, 
                horizontalalignment='center', color=color)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Feature Value')
    ax.set_title('Temporal Segments')
    ax.legend()
    
    plt.tight_layout()
    return fig


# Example usage:
def example_usage():
    """Example usage of the temporal alignment and classification functions."""
    # Create sample sequence
    seq_len = 100
    embed_dim = 64
    sequence = torch.randn(seq_len, embed_dim)
    
    # Create temporal aligner
    aligner = TemporalAlignment(method='dtw', n_segments=5)
    
    # Align sequence
    segments, segment_embeddings = aligner.align(sequence)
    
    print(f"Found {len(segments)} segments:")
    for i, (start, end) in enumerate(segments):
        print(f"Segment {i}: frames {start}-{end} (length: {end-start+1})")
    
    # Create sample classifier and encoder for demonstration
    class DummyClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            # Just return the input as embedding for this example
            return x, None
    
    # Initialize dummy models
    classifier = DummyClassifier(embed_dim, 10)
    encoder = DummyEncoder()
    
    # Create temporal classifier
    temp_classifier = TemporalClassifier(classifier, encoder)
    
    # Classify segments
    labels, confidences = temp_classifier.classify_segments(segments, segment_embeddings)
    
    print("Classification results:")
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        print(f"Segment {i}: class {label} (confidence: {conf:.2f})")
    
    # Visualize segments
    fig = visualize_segments(sequence, segments, labels, confidences)
    fig.savefig('temporal_segments.png')
    print("Visualization saved to 'temporal_segments.png'")


if __name__ == "__main__":
    example_usage()