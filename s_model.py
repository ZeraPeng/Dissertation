import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def fuse_logits(part_logits_list, fusion_type="weighted_sum"):
    """
    Fuse global logits and six part logits using different fusion strategies.

    Args:
        part_logits_list (list of Tensor): A list of 6 tensors, each of shape [batch_size, num_classes].
        fusion_type (str): Fusion method, choose from:
            - "weighted_sum" (default): Weighted sum fusion.
            - "logsumexp": Log-Sum-Exp fusion for stability.

    Returns:
        fused_logits (Tensor): Shape [batch_size, num_classes], fused logits.
    """
    # assert len(part_logits_list) == 6, "part_logits_list must contain exactly 6 tensors."

    # Compute the mean of part logits
    part_logits = sum(part_logits_list) / len(part_logits_list)  # Shape: [batch_size, num_classes]

    if fusion_type == "weighted_sum":
        # ⚡ Method 1: Weighted sum fusion
        fused_logits = part_logits
    elif fusion_type == "logsumexp":
        # ⚡ Method 2: Log-Sum-Exp fusion (more numerically stable)
        stacked_logits = torch.stack(part_logits_list, dim=0)  # Shape: [7, batch_size, num_classes]
        fused_logits = torch.logsumexp(stacked_logits, dim=0)  # Log-Sum-Exp computation
    else:
        raise ValueError("Unsupported fusion_type. Use 'weighted_sum' or 'logsumexp'.")

    return fused_logits


class Encoder(nn.Module):
    def __init__(self, layer_sizes, style_latent_size=0):
        super(Encoder, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))  # layer_sizes: vis_emb_input_size, semantic_latent_size + style_latent_size
            layers.append(nn.Dropout1d())
            layers.append(nn.ReLU())

        self.style_latent_size = style_latent_size

        self.model = nn.Sequential(*layers) # pooling
        self.mu = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        self.logvar = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )

        self.apply(weights_init)

    def forward(self, x, instance_style=False, type='global'):

        h = self.model(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.style_latent_size == 0:
            return mu, logvar

        if not instance_style:
            return (
                mu[:, :-self.style_latent_size],
                logvar[:, :-self.style_latent_size]
            )
        else:
            return (
                mu[:, :-self.style_latent_size],
                logvar[:, :-self.style_latent_size],
                mu[:, -self.style_latent_size:],
                logvar[:, -self.style_latent_size:]
            )


class Decoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, x):

        out = self.model(x)
        return out


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))  # layer_sizes: [semantic_latent_size, ss]
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)

class MLP2(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP2, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))  # layer_sizes: [semantic_latent_size, ss]
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape  # (2500, 7, 96)
        x = x.view(-1, feat_dim)  # (2500 * 7, 96)
        x = self.model(x)  # (2500 * 7, 5)
        x = x.view(batch_size, seq_len, -1)  # (2500, 7, 5)
        # x, _ = x.max(dim=1)  #  maxpool, (2500, 5)
        x = x.mean(dim=1)  #  maxpool, (2500, 5)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size//4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size//4, 1),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.net(x)


def reparameterize(mu, logvar):     # VAE reparameterize. mu: mean, logvar: log varianc -> generate random samples
    sigma = torch.exp(0.5*logvar)
    eps = torch.FloatTensor(sigma.size()[0], 1).normal_(
        0, 1).expand(sigma.size()).to(mu.device)
    return eps*sigma + mu


def KL_divergence(mu, logvar):
    return 0.5*(torch.sum(- (mu**2) + 1 + logvar - torch.exp(logvar)))/mu.shape[0]


def permute_dims(zs, zis):
    B = zs.size(0)
    device = zs.device
    perm1 = torch.randperm(B, device=device)
    perm2 = torch.randperm(B, device=device)

    perm_zs = zs[perm1]
    perm_zis = zis[perm2]

    return perm_zs, perm_zis


class ZeroShotClassifier(nn.Module):
    """
    Classifier that samples frames from a sequence and predicts top-k sub-actions/objects
    from each frame, then maps to unseen class labels through a probability graph.
    """
    def __init__(self, input_dim, output_dim, num_frames=4, top_k=5):
        """
        Args:
            input_dim (int): Dimension of the input features
            output_dim (int): Number of output classes
            num_frames (int): Number of frames to sample from the sequence
            top_k (int): Number of top sub-actions/objects to consider per frame
        """
        super(ZeroShotClassifier, self).__init__()
        self.num_frames = num_frames
        self.top_k = top_k
        
        # Base classifier for predicting sub-actions & objects
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )
        
        # Mapping network to unseen classes (implemented as fully-connected layer)
        self.mapping = nn.Linear(output_dim * num_frames, output_dim)
        
    def sample_frames(self, x):
        """
        Sample frames from a sequence
        
        Args:
            x (torch.Tensor): Input sequences of shape [batch_size, seq_len, feat_dim]
            
        Returns:
            torch.Tensor: Sampled frames of shape [batch_size, num_frames, feat_dim]
        """
        batch_size, seq_len, feat_dim = x.shape
        
        if seq_len <= self.num_frames:
            # If sequence is shorter than requested frames, pad with duplicates
            indices = torch.arange(seq_len, device=x.device)
            if seq_len < self.num_frames:
                padding = torch.tensor([seq_len-1] * (self.num_frames - seq_len), 
                                      device=x.device)
                indices = torch.cat([indices, padding])
        else:
            # Evenly sample frames across the sequence
            indices = torch.linspace(0, seq_len-1, self.num_frames, dtype=torch.long, device=x.device)
        
        return torch.index_select(x, 1, indices)
    
    def forward(self, sequence_encoder, x, unseen_inds=None):
        """
        Forward pass through the classifier
        
        Args:
            sequence_encoder: Encoder model to extract features from frames
            x (torch.Tensor): Input sequences [batch size, embedding size, frames]
            unseen_inds: Indices of unseen classes for mapping
            
        Returns:
            tuple: (frame_logits, class_logits, frame_predictions)
        """
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        # Handle different input formats
        if len(x.shape) == 3:  # [batch, seq_len, feat_dim]
            sampled_frames = self.sample_frames(x)
        else:  # [batch, feat_dim] - treat as single frame
            sampled_frames = x.unsqueeze(1).repeat(1, self.num_frames, 1)
        
        frame_logits = []
        frame_features = []
        
        # Process each sampled frame
        for i in range(self.num_frames):
            frame = sampled_frames[:, i]
            
            # Extract features using the sequence encoder
            with torch.no_grad():
                frame_feat, _ = sequence_encoder(frame)
            
            frame_features.append(frame_feat)
            
            # Predict sub-actions & objects for this frame
            logits = self.classifier(frame_feat)
            frame_logits.append(logits)
        
        # Stack frame logits [batch, num_frames, num_classes]
        frame_logits = torch.stack(frame_logits, dim=1)
        
        # Get top-k predictions for each frame
        probs = F.softmax(frame_logits, dim=-1)
        topk_values, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Create probability graph by flattening across frames
        # [batch, num_frames * num_classes]
        flat_probs = probs.reshape(batch_size, -1)
        
        # Map to unseen class labels
        class_logits = self.mapping(flat_probs)
        
        if unseen_inds is not None:
            # Map to unseen classes
            class_logits = class_logits[:, unseen_inds]
        
        return frame_logits, class_logits, (topk_values, topk_indices)
    
    def predict(self, sequence_encoder, x, unseen_inds=None):
        """
        Make predictions for input sequences
        
        Args:
            sequence_encoder: Encoder model
            x (torch.Tensor): Input sequences
            unseen_inds: Indices of unseen classes
            
        Returns:
            torch.Tensor: Final class predictions
        """
        _, class_logits, _ = self.forward(sequence_encoder, x, unseen_inds)
        predictions = torch.argmax(class_logits, dim=-1)
        
        if unseen_inds is not None:
            # Map predictions back to original class indices
            predictions = torch.tensor(unseen_inds, device=predictions.device)[predictions]
            
        return predictions