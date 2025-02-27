import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
import torch
import torch.nn.functional as F

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
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
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
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


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
