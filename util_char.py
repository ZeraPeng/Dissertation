import numpy as np
import torch
import torch.nn.functional as F

def compute_alignment_score(visual_features, text_features):
    """
    Compute alignment scores between pooled visual features and text features.
    
    Args:
        visual_features: tensor of shape (batch_size, feature_dim)
            The pooled visual features across frames.
        text_features: tensor of shape (num_classes, num_prompts, feature_dim)
            The text features for each class and prompt.
            
    Returns:
        alignment_scores: tensor of shape (batch_size, num_classes)
            The overall alignment scores between visuals and classes.
    """
    batch_size = visual_features.shape[0]
    num_classes = text_features.shape[0]
    num_prompts = text_features.shape[1]
    
    # Normalize features for cosine similarity
    visual_features = F.normalize(visual_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=2)
    
    # Initialize alignment scores
    v2t_scores = torch.zeros((batch_size, num_classes), device=visual_features.device)
    t2v_scores = torch.zeros((batch_size, num_classes), device=visual_features.device)
    
    # Calculate visual-to-text alignment scores
    for n in range(batch_size):
        for k in range(num_classes):
            # Compute similarities between visual n and all prompts of class k
            similarities = torch.matmul(visual_features[n], text_features[k].t())
            # Take the maximum similarity across prompts
            v2t_scores[n, k] = torch.max(similarities)
    
    # Calculate text-to-visual alignment scores
    for n in range(batch_size):
        for k in range(num_classes):
            # Compute average similarity across all prompts
            similarities = torch.matmul(visual_features[n], text_features[k].t())
            t2v_scores[n, k] = torch.mean(similarities)
    
    # Calculate overall alignment scores
    alignment_scores = (v2t_scores + t2v_scores) / 2
    
    return alignment_scores


def compute_simplified_alignment_score(visual_features, text_features):
    """
    Compute simplified alignment scores between pooled visual features and pooled text features.
    
    Args:
        visual_features: tensor of shape (batch_size, feature_dim)
            The pooled visual features across frames.
        text_features: tensor of shape (num_classes, num_prompts, feature_dim)
            The text features for each class and prompt.
            
    Returns:
        alignment_scores: tensor of shape (batch_size, num_classes)
            The simplified alignment scores between visuals and classes.
    """
    batch_size = visual_features.shape[0]
    num_classes = text_features.shape[0]
    
    # Normalize features for cosine similarity
    visual_features = F.normalize(visual_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=2)
    
    # Pool text features across prompts for each class
    pooled_text_features = torch.mean(text_features, dim=1)  # Shape: (num_classes, feature_dim)
    pooled_text_features = F.normalize(pooled_text_features, p=2, dim=1)
    
    # Compute dot product between visual features and pooled text features
    alignment_scores = torch.matmul(visual_features, pooled_text_features.t())  # Shape: (batch_size, num_classes)
    
    return alignment_scores

def compute_cross_entropy_loss(alignment_scores, labels):
    """
    Compute cross-entropy loss from alignment scores.
    
    Args:
        alignment_scores: tensor of shape (batch_size, num_classes)
            The alignment scores between videos and classes.
        labels: tensor of shape (batch_size)
            The ground truth class indices for each video.
            
    Returns:
        loss: scalar tensor
            The cross-entropy loss.
    """
    loss = F.cross_entropy(alignment_scores, labels)
    
    return loss