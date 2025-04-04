import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import imageio

# If using Colab
from google.colab import drive
drive.mount('/content/drive')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Transforms for CelebA (64x64)
# =============================================================================
transform_train = transforms.Compose([
    transforms.CenterCrop(178),               # Crop to the face region (e.g. 178x178)
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

batch_size = 128  # adjust based on available GPU memory

# =============================================================================
# Class names for the binary attribute ("Smiling")
# =============================================================================
class_names = ["Not Smiling", "Smiling"]

# =============================================================================
# Custom CelebA Dataset for Smiling attribute
# =============================================================================
class CelebASmiling(datasets.CelebA):
    def __init__(self, root, split, transform, download=True):
        # Use target_type "attr" to get the attributes
        super().__init__(root=root, split=split, target_type="attr", transform=transform, download=download)
        # Get index for the "Smiling" attribute from the attribute names list
        self.smiling_idx = self.attr_names.index("Smiling")

    def __getitem__(self, index):
        img, attr = super().__getitem__(index)
        # Convert the attribute value from (-1, 1) to (0, 1)
        label = 1 if attr[self.smiling_idx] == 1 else 0
        return img, label

# =============================================================================
# Activation and Helper Layers
# =============================================================================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            Swish(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention

class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=256, min_distance=1.0, repulsion_strength=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.min_distance = min_distance
        self.repulsion_strength = repulsion_strength
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

        with torch.no_grad():
            if num_classes == 2:
                self.centers[0] = -torch.ones(feat_dim) / math.sqrt(feat_dim)
                self.centers[1] = torch.ones(feat_dim) / math.sqrt(feat_dim)
            else:
                for i in range(num_classes):
                    self.centers[i] = torch.randn(feat_dim)
                    self.centers[i] = self.centers[i] / torch.norm(self.centers[i]) * 2.0

    def compute_pairwise_distances(self, x, y):
        n = x.size(0)
        m = y.size(0)
        x_norm = (x ** 2).sum(1).view(n, 1)
        y_norm = (y ** 2).sum(1).view(1, m)
        distmat = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
        distmat = torch.clamp(distmat, min=1e-12)
        distmat = torch.sqrt(distmat)
        return distmat

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = self.compute_pairwise_distances(x, self.centers)
        classes = torch.arange(self.num_classes).to(labels.device)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))
        attraction_dist = distmat * mask.float()
        attraction_loss = attraction_dist.sum() / batch_size
        center_distances = self.compute_pairwise_distances(self.centers, self.centers)
        diff_mask = 1.0 - torch.eye(self.num_classes, device=x.device)
        repulsion_loss = torch.clamp(self.min_distance - center_distances, min=0.0)
        repulsion_loss = (repulsion_loss * diff_mask).sum() / (self.num_classes * (self.num_classes - 1) + 1e-6)
        intra_class_variance = 0.0
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if torch.sum(class_mask) > 1:
                class_samples = x[class_mask]
                class_center = torch.mean(class_samples, dim=0)
                variance = torch.mean(torch.sum((class_samples - class_center) ** 2, dim=1))
                intra_class_variance += variance
        if self.num_classes > 0:
            intra_class_variance = intra_class_variance / self.num_classes
        total_loss = attraction_loss + self.repulsion_strength * repulsion_loss - 0.1 * intra_class_variance
        with torch.no_grad():
            self.avg_center_dist = torch.sum(center_distances * diff_mask) / (
                        self.num_classes * (self.num_classes - 1) + 1e-6)
            self.avg_sample_dist = torch.mean(distmat)
            self.center_attraction = attraction_loss.item()
            self.center_repulsion = repulsion_loss.item()
            self.intra_variance = intra_class_variance.item() if isinstance(intra_class_variance,
                                                                            torch.Tensor) else intra_class_variance
        return total_loss

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var  = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x

# =============================================================================
# Residual Block (unchanged except for potential extra modules)
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln1   = LayerNorm2d(channels)
        self.swish = Swish()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ln2   = LayerNorm2d(channels)
        self.ca    = CALayer(channels)
        self.sa    = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.swish(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        out += residual
        out = self.swish(out)
        return out

# =============================================================================
# Updated Encoder for 64x64 CelebA
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # 64×64
            LayerNorm2d(64),
            Swish()
        )
        self.skip_features = []
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64×64 -> 32×32
            LayerNorm2d(128),
            Swish()
        )
        self.res1 = ResidualBlock(128)
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32×32 -> 16×16
            LayerNorm2d(256),
            Swish()
        )
        self.res2 = ResidualBlock(256)
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16×16 -> 8×8
            LayerNorm2d(512),
            Swish()
        )
        self.res3 = ResidualBlock(512)
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),  # 8×8 -> 4×4
            LayerNorm2d(1024),
            Swish()
        )
        self.res4 = ResidualBlock(1024)
        # Project to latent space from 4×4 feature map
        self.fc_mu = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        self.skip_features = []
        x = self.initial_conv(x)
        self.skip_features.append(x)
        x = self.down1(x)
        x = self.res1(x)
        self.skip_features.append(x)
        x = self.down2(x)
        x = self.res2(x)
        self.skip_features.append(x)
        x = self.down3(x)
        x = self.res3(x)
        self.skip_features.append(x)
        x = self.down4(x)
        x = self.res4(x)
        self.skip_features.append(x)
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

# =============================================================================
# Updated Decoder for 64x64 CelebA
# =============================================================================
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        # Project latent vector to 4×4 feature map with 1024 channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            Swish(),
            nn.Linear(1024, 1024 * 4 * 4),
            nn.LayerNorm(1024 * 4 * 4),
            Swish()
        )
        self.res4 = ResidualBlock(1024)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),  # 4×4 -> 8×8
            nn.GroupNorm(32, 512),
            Swish()
        )
        self.res3 = ResidualBlock(512)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 8×8 -> 16×16
            nn.GroupNorm(16, 256),
            Swish()
        )
        self.res2 = ResidualBlock(256)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16×16 -> 32×32
            nn.GroupNorm(8, 128),
            Swish()
        )
        self.res1 = ResidualBlock(128)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32×32 -> 64×64
            nn.GroupNorm(8, 64),
            Swish()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            Swish(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, encoder_features=None):
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.res4(x)
        x = self.up4(x)
        x = self.res3(x)
        x = self.up3(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res1(x)
        x = self.up1(x)
        x = self.final_conv(x)
        return x

# =============================================================================
# Euclidean Distance Loss
# =============================================================================
def euclidean_distance_loss(x, y, reduction='mean'):
    squared_diff = (x - y) ** 2
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)
    euclidean_dist = torch.sqrt(squared_dist + 1e-8)
    if reduction == 'mean':
        return euclidean_dist.mean()
    elif reduction == 'sum':
        return euclidean_dist.sum()
    else:
        return euclidean_dist

# =============================================================================
# Simple Autoencoder that combines the updated Encoder and Decoder
# =============================================================================
class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_classes=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            Swish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.register_buffer('class_centers', torch.zeros(num_classes, latent_dim))
        self.register_buffer('center_counts', torch.zeros(num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def encode_with_params(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        return mu, logvar

    def decode(self, z):
        encoder_features = getattr(self, 'stored_encoder_features', None)
        return self.decoder(z, encoder_features)

    def classify(self, z):
        return self.classifier(z)

    def compute_center_loss(self, z, labels):
        centers_batch = self.class_centers[labels]
        squared_diff = (z - centers_batch) ** 2
        squared_dist = squared_diff.sum(dim=1)
        center_loss = torch.sqrt(squared_dist + 1e-8).mean()
        return center_loss

    def update_centers(self, z, labels, momentum=0.9):
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_samples = z[mask]
                class_mean = class_samples.mean(0)
                old_center = self.class_centers[label]
                new_center = momentum * old_center + (1 - momentum) * class_mean
                self.class_centers[label] = new_center

    def kl_divergence(self, mu, logvar):
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.clamp(kl_loss, min=0.0, max=100.0).mean()
        mu_reg = 1e-4 * torch.sum(mu.pow(2))
        return kl_loss + mu_reg

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.stored_encoder_features = self.encoder.skip_features
        x_recon = self.decoder(z, self.encoder.skip_features)
        return x_recon, mu, logvar, z

# =============================================================================
# Additional Modules for Diffusion Model
# =============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels=256):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels, self.n_channels * 2)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels * 2, self.n_channels)

    def forward(self, t):
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if emb.shape[1] < self.n_channels:
            padding = torch.zeros(emb.shape[0], self.n_channels - emb.shape[1], device=emb.device)
            emb = torch.cat([emb, padding], dim=1)
        return self.lin2(self.act(self.lin1(emb)))

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, n_channels=256):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, n_channels)
        self.lin1 = nn.Linear(n_channels, n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, c):
        emb = self.embedding(c)
        return self.lin2(self.act(self.lin1(emb)))

class UNetAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 2, 3)
        v = v.permute(0, 1, 3, 2)
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 3, 1, 2)
        out = out.reshape(b, c, h, w)
        output = self.proj(out) + residual
        return output

class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time=256, dropout_rate=0.2):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(d_time, out_channels)
        self.class_emb = nn.Linear(d_time, out_channels)
        self.act = Swish()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t, c=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        t_emb = self.act(self.time_emb(t))
        h = h + t_emb.view(-1, t_emb.shape[1], 1, 1)
        if c is not None:
            c_emb = self.act(self.class_emb(c))
            h = h + c_emb.view(-1, c_emb.shape[1], 1, 1)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.residual(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, t=None, c=None):
        for layer in self:
            if isinstance(layer, UNetResidualBlock):
                x = layer(x, t, c)
            elif isinstance(layer, UNetAttentionBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x

# =============================================================================
# Updated Conditional UNet (Convolutional version) for z-space Diffusion Model
# =============================================================================
class ConditionalUNet(nn.Module):
    def __init__(self, latent_dim=256, time_emb_dim=256, num_classes=2, base_channels=64):
        """
        This convolutional UNet operates on a 2D feature map representation of the latent code.
        Here we reshape the latent vector of shape (B, 256) into a feature map of shape (B, 4, 8, 8)
        (since 4*8*8 = 256). The network then applies two encoder blocks, a bottleneck, and two decoder
        blocks while incorporating conditioning via time and class embeddings.
        """
        super().__init__()
        # Set the target 2D dimensions; here we choose 8x8 with 4 channels (4*8*8 = 256)
        self.H = 8
        self.W = 8
        self.in_channels = latent_dim // (self.H * self.W)  # should be 256 // 64 = 4
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        
        # Time and class embeddings for conditioning
        self.time_embedding = TimeEmbedding(n_channels=time_emb_dim)
        self.class_embedding = ClassEmbedding(num_classes=num_classes, n_channels=time_emb_dim)
        
        # Project latent vector to a feature map of shape (B, 4, 8, 8)
        self.project_in = nn.Linear(latent_dim, self.in_channels * self.H * self.W)
        
        # -------------------- Encoder --------------------
        # Encoder Block 1: keeps resolution at 8x8
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            Swish(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            Swish()
        )
        # Conditioning for encoder block 1
        self.time_proj_enc1 = nn.Linear(time_emb_dim, base_channels)
        self.class_proj_enc1 = nn.Linear(time_emb_dim, base_channels)
        
        # Encoder Block 2: downsample from 8x8 to 4x4
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            Swish(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            Swish()
        )
        self.time_proj_enc2 = nn.Linear(time_emb_dim, base_channels * 2)
        self.class_proj_enc2 = nn.Linear(time_emb_dim, base_channels * 2)
        
        # -------------------- Bottleneck --------------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            Swish(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            Swish()
        )
        self.time_proj_bottleneck = nn.Linear(time_emb_dim, base_channels * 2)
        self.class_proj_bottleneck = nn.Linear(time_emb_dim, base_channels * 2)
        
        # -------------------- Decoder --------------------
        # Decoder Block 1: upsample from 4x4 back to 8x8
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            Swish()
        )
        self.time_proj_dec1 = nn.Linear(time_emb_dim, base_channels)
        self.class_proj_dec1 = nn.Linear(time_emb_dim, base_channels)
        
        # Decoder Block 2: refine features at 8x8 resolution
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels, self.in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.in_channels),
            Swish(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.in_channels),
            Swish()
        )
        
        # Project the refined feature map back to a latent vector
        self.project_out = nn.Linear(self.in_channels * self.H * self.W, latent_dim)
    
    def forward(self, z, t, c):
        """
        z: latent vector of shape (B, latent_dim)
        t: time-step tensor of shape (B,) or (B, 1)
        c: class label tensor of shape (B,)
        """
        batch_size = z.size(0)
        
        # Compute time and class conditioning embeddings
        t_emb = self.time_embedding(t)  # (B, time_emb_dim)
        c_emb = self.class_embedding(c)   # (B, time_emb_dim)
        
        # Project the latent vector to a 2D feature map
        x = self.project_in(z)  # (B, in_channels * H * W)
        x = x.view(batch_size, self.in_channels, self.H, self.W)  # (B, 4, 8, 8)
        
        # --- Encoder Block 1 ---
        x = self.enc1(x)  # (B, base_channels, 8, 8)
        cond1 = self.time_proj_enc1(t_emb) + self.class_proj_enc1(c_emb)  # (B, base_channels)
        cond1 = cond1.view(batch_size, -1, 1, 1)
        x = x + cond1
        
        # --- Encoder Block 2 (Downsampling) ---
        x = self.enc2(x)  # (B, base_channels*2, 4, 4)
        cond2 = self.time_proj_enc2(t_emb) + self.class_proj_enc2(c_emb)  # (B, base_channels*2)
        cond2 = cond2.view(batch_size, -1, 1, 1)
        x = x + cond2
        
        # --- Bottleneck ---
        x = self.bottleneck(x)  # (B, base_channels*2, 4, 4)
        cond_b = self.time_proj_bottleneck(t_emb) + self.class_proj_bottleneck(c_emb)
        cond_b = cond_b.view(batch_size, -1, 1, 1)
        x = x + cond_b
        
        # --- Decoder Block 1 (Upsampling) ---
        x = self.dec1(x)  # (B, base_channels, 8, 8)
        cond_d1 = self.time_proj_dec1(t_emb) + self.class_proj_dec1(c_emb)
        cond_d1 = cond_d1.view(batch_size, -1, 1, 1)
        x = x + cond_d1
        
        # --- Decoder Block 2 (Refinement) ---
        x = self.dec2(x)  # (B, in_channels, 8, 8)
        
        # Project back to the original latent vector shape
        x = x.view(batch_size, -1)  # (B, in_channels*H*W)
        out = self.project_out(x)   # (B, latent_dim)
        return out

# =============================================================================
# Conditional Denoise Diffusion Module (remains unchanged)
# =============================================================================
class ConditionalDenoiseDiffusion():
    def __init__(self, eps_model, n_steps=1000, device=None):
        super().__init__()
        self.eps_model = eps_model
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    def p_sample(self, xt, t, c=None):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)
        eps_theta = self.eps_model(xt, t, c)
        alpha_t = self.alpha[t].reshape(-1, 1).to(xt.device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1).to(xt.device)
        mean = (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta) / torch.sqrt(alpha_t)
        var = self.beta[t].reshape(-1, 1)
        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    def sample(self, shape, device, c=None):
        x = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t, c)
        return x

    def loss(self, x0, labels=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        eps_theta = self.eps_model(xt, t, labels)
        return euclidean_distance_loss(eps, eps_theta)

# =============================================================================
# Visualization Functions
# =============================================================================
def generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    n_classes_vis = len(class_names)
    fig, axes = plt.subplots(n_classes_vis, n_per_class + 1, figsize=((n_per_class + 1) * 2, n_classes_vis * 2))
    fig.suptitle('CelebA (Smiling) Samples Generated by VAE-Diffusion Model', fontsize=16, y=0.98)
    for i in range(n_classes_vis):
        axes[i, 0].text(0.5, 0.5, class_names[i],
                        fontsize=10, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        axes[i, 0].axis('off')
        class_tensor = torch.tensor([i] * n_per_class, device=device)
        latent_shape = (n_per_class, autoencoder.latent_dim)
        samples = diffusion.sample(latent_shape, device, class_tensor)
        with torch.no_grad():
            decoded = autoencoder.decode(samples)
        for j in range(n_per_class):
            img = decoded[j].cpu().permute(1, 2, 0).numpy()
            axes[i, j + 1].imshow(img)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f'Sample {j + 1}', fontsize=9)
    description = (
        "This visualization shows images generated by the conditional diffusion model using a VAE decoder on CelebA.\n"
        "Both classes (Smiling vs Not Smiling) are visualized."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_path = f"{save_dir}/vae_samples_grid_celeba.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    autoencoder.train()
    diffusion.eps_model.train()
    print("Generated sample grid for CelebA classes")
    return save_path

def visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=None):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    print(f"Generating latent space projection for class {class_names[class_idx]}...")
    test_dataset = CelebASmiling(root="./data", split="test", transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, logvar = autoencoder.encode_with_params(images)
            all_latents.append(mu.detach().cpu().numpy())
            all_labels.append(labels.numpy())
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(all_latents)
    n_samples = 5
    steps_to_show = 8
    step_size = diffusion.n_steps // steps_to_show
    timesteps = list(range(0, diffusion.n_steps, step_size))[::-1]
    class_tensor = torch.tensor([class_idx] * n_samples, device=device)
    x = torch.randn((n_samples, autoencoder.latent_dim), device=device)
    samples_per_step = []
    path_latents = []
    with torch.no_grad():
        for t in timesteps:
            current_x = x.clone()
            for time_step in range(t, -1, -1):
                current_x = diffusion.p_sample(current_x, torch.tensor([time_step], device=device), class_tensor)
            path_latents.append(current_x[0:1].detach().cpu().numpy())
            decoded = autoencoder.decode(current_x)
            samples_per_step.append(decoded.cpu())
        path_latents.append(current_x[0:1].detach().cpu().numpy())
    path_latents = np.vstack(path_latents)
    path_2d = pca.transform(path_latents)
    fig = plt.figure(figsize=(16, 16))
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)
    ax_denoising = fig.add_subplot(gs[0])
    grid_rows = n_samples
    grid_cols = len(timesteps)
    ax_denoising.set_title(f"VAE-Diffusion Denoising Process for {class_names[class_idx]}", fontsize=16, pad=10)
    ax_denoising.set_xticks([])
    ax_denoising.set_yticks([])
    gs_denoising = gs[0].subgridspec(grid_rows, grid_cols, wspace=0.1, hspace=0.1)
    for i in range(n_samples):
        for j, t in enumerate(timesteps):
            ax = fig.add_subplot(gs_denoising[i, j])
            img = samples_per_step[j][i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            if i == 0:
                ax.set_title(f't={t}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Sample {i + 1}", fontsize=9)
            if i == 0:
                for spine in ax.spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(2)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.figtext(0.02, 0.65, "Path Tracked →", fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))
    ax_latent = fig.add_subplot(gs[1])
    for i in range(len(class_names)):
        mask = all_labels == i
        alpha = 0.3 if i != class_idx else 0.8
        size = 20 if i != class_idx else 40
        ax_latent.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            label=class_names[i],
            alpha=alpha,
            s=size
        )
    ax_latent.plot(
        path_2d[:, 0],
        path_2d[:, 1],
        'r-o',
        linewidth=2.5,
        markersize=8,
        label="Diffusion Path",
        zorder=10
    )
    for i in range(len(path_2d) - 1):
        ax_latent.annotate(
            "",
            xy=(path_2d[i + 1, 0], path_2d[i + 1, 1]),
            xytext=(path_2d[i, 0], path_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5)
        )
    for i, t in enumerate(timesteps):
        ax_latent.annotate(
            f"t={t}",
            xy=(path_2d[i, 0], path_2d[i, 1]),
            xytext=(path_2d[i, 0] + 2, path_2d[i, 1] + 2),
            fontsize=8,
            color='darkred'
        )
    ax_latent.scatter(path_2d[0, 0], path_2d[0, 1], c='black', s=100, marker='x', label="Start (Noise)", zorder=11)
    ax_latent.scatter(path_2d[-1, 0], path_2d[-1, 1], c='green', s=100, marker='*', label="End (Generated)", zorder=11)
    target_mask = all_labels == class_idx
    target_center = np.mean(latents_2d[target_mask], axis=0)
    ax_latent.scatter(target_center[0], target_center[1], c='green', s=300, marker='*',
                      edgecolor='black', alpha=0.7, zorder=9)
    ax_latent.annotate(
        f"TARGET: {class_names[class_idx]}",
        xy=(target_center[0], target_center[1]),
        xytext=(target_center[0] + 5, target_center[1] + 5),
        fontsize=14,
        fontweight='bold',
        color='darkgreen',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    ax_latent.set_title(f"VAE-Diffusion Path in Latent Space for {class_names[class_idx]}", fontsize=16)
    ax_latent.legend(fontsize=10, loc='best')
    ax_latent.grid(True, linestyle='--', alpha=0.7)
    plt.figtext(
        0.5, 0.01,
        "This visualization shows the denoising process (top) and the corresponding path in latent space (bottom).\n"
        "The first row (highlighted in red) corresponds to the latent space path.",
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
    if save_path is None:
        save_path = f"./results/denoising_path_{class_names[class_idx]}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VAE Denoising visualization for {class_names[class_idx]} saved to {save_path}")
    autoencoder.train()
    diffusion.eps_model.train()
    return save_path

def visualize_reconstructions(autoencoder, epoch, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    test_dataset = CelebASmiling(root="./data", split="test", transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    autoencoder.eval()
    with torch.no_grad():
        mu, logvar = autoencoder.encode_with_params(test_images)
        z = autoencoder.reparameterize(mu, logvar)
        reconstructed = autoencoder.decode(z)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        img = test_images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        label_text = class_names[test_labels[i]] if class_names is not None else str(test_labels[i].item())
        axes[0, i].set_title(f'Original: {label_text}')
        axes[0, i].axis('off')
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_reconstruction_epoch_{epoch}.png")
    plt.close()
    autoencoder.train()

def visualize_latent_space(autoencoder, epoch, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device
    test_dataset = CelebASmiling(root="./data", split="test", transform=transform_test, download=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    autoencoder.eval()
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            mu, logvar = autoencoder.encode_with_params(images)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
        latents_2d = tsne.fit_transform(all_latents)
        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):
            mask = all_labels == i
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], label=class_names[i], alpha=0.6)
        plt.title(f"t-SNE Visualization of VAE Latent Space (Epoch {epoch})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vae_latent_space_epoch_{epoch}.png")
        plt.close()
    except Exception as e:
        print(f"t-SNE visualization error: {e}")
    autoencoder.train()

def generate_class_samples(autoencoder, diffusion, target_class, num_samples=5, save_path=None):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    if isinstance(target_class, str):
        if target_class in class_names:
            target_class = class_names.index(target_class)
        else:
            raise ValueError(f"Invalid class name: {target_class}. Must be one of {class_names}")
    class_tensor = torch.tensor([target_class] * num_samples, device=device)
    latent_shape = (num_samples, autoencoder.latent_dim)
    with torch.no_grad():
        latent_samples = diffusion.sample(latent_shape, device, class_tensor)
        samples = autoencoder.decode(latent_samples)
    if save_path:
        plt.figure(figsize=(num_samples * 2, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{class_names[target_class]}")
        plt.suptitle(f"Generated {class_names[target_class]} Samples")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return samples

def create_diffusion_animation(autoencoder, diffusion, class_idx, num_frames=50, seed=42,
                               save_path=None, temp_dir=None, fps=10, reverse=False):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    if isinstance(class_idx, str):
        if class_idx in class_names:
            class_idx = class_names.index(class_idx)
        else:
            raise ValueError(f"Invalid class name: {class_idx}. Must be one of {class_names}")
    if temp_dir is None:
        temp_dir = os.path.join('./temp_frames', f'class_{class_idx}_{seed}')
    os.makedirs('./temp_frames', exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    if save_path is None:
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'diffusion_animation_{class_names[class_idx]}.gif')
    torch.manual_seed(seed)
    np.random.seed(seed)
    class_tensor = torch.tensor([class_idx], device=device)
    total_steps = diffusion.n_steps
    if num_frames >= total_steps:
        timesteps = list(range(total_steps))
    else:
        step_size = total_steps // num_frames
        timesteps = list(range(0, total_steps, step_size))
        if timesteps[-1] != total_steps - 1:
            timesteps.append(total_steps - 1)
    if reverse:
        timesteps = sorted(timesteps, reverse=True)
    else:
        timesteps = sorted(timesteps)
        backward_timesteps = sorted(timesteps[1:-1], reverse=True)
        timesteps = timesteps + backward_timesteps
    print(f"Creating diffusion animation for class '{class_names[class_idx]}'...")
    frame_paths = []
    with torch.no_grad():
        print("Generating initial clean image...")
        x = torch.randn((1, autoencoder.latent_dim), device=device)
        for time_step in tqdm(range(total_steps - 1, -1, -1), desc="Denoising"):
            x = diffusion.p_sample(x, torch.tensor([time_step], device=device), class_tensor)
        clean_x = x.clone()
        print("Generating animation frames...")
        for i, t in enumerate(timesteps):
            current_x = clean_x.clone()
            if t > 0:
                torch.manual_seed(seed)
                eps = torch.randn_like(current_x)
                alpha_bar_t = diffusion.alpha_bar[t].reshape(-1, 1)
                current_x = torch.sqrt(alpha_bar_t) * current_x + torch.sqrt(1 - alpha_bar_t) * eps
            decoded = autoencoder.decode(current_x)
            img = decoded[0].cpu().permute(1, 2, 0).numpy()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img)
            ax.axis('off')
            progress = (t / total_steps) * 100
            title = f'Class: {class_names[class_idx]} (t={t}, {progress:.1f}% noise)'
            ax.set_title(title)
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_path, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)
    print(f"Creating GIF animation at {fps} fps...")
    with imageio.get_writer(save_path, mode='I', fps=fps, loop=0) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    print("Cleaning up temporary files...")
    for frame_path in frame_paths:
        os.remove(frame_path)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass
    print(f"Animation saved to {save_path}")
    return save_path

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.feature_extractor = vgg
        self.criterion = euclidean_distance_loss
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x, y):
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return self.criterion(x_features, y_features)

class Discriminator64(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),  # 64×64 -> 32×32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),            # 32×32 -> 16×16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),           # 16×16 -> 8×8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),           # 8×8 -> 4×4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 2),                                  # output: (B, 1, 3, 3)
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)  # out has shape (B, 1, 3, 3)
        # Average over spatial dimensions so each sample is a single scalar
        out = out.mean(dim=[2, 3])  # shape becomes (B, 1)
        return out.view(-1)        # now returns a tensor of shape (B,)


# =============================================================================
# Training Functions
# =============================================================================
def train_autoencoder(autoencoder, train_loader, num_epochs=300, lr=1e-4,
                      lambda_cls=0.1, lambda_center=0.05, lambda_vgg=0.4, lambda_gan=0.2,
                      kl_weight_start=0.001, kl_weight_end=0.05,
                      visualize_every=10, save_dir="./results"):
    print("Starting VAE-GAN training with perceptual loss enhancement...")
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    vgg_loss = VGGPerceptualLoss(device)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    discriminator = Discriminator64().to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    gan_criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=num_epochs * len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    loss_history = {'total': [], 'recon': [], 'kl': [], 'class': [], 'center': [], 'perceptual': [], 'gan': []}
    best_loss = float('inf')
    lambda_recon = 1.0

    # Initialize GradScaler for automatic mixed precision (AMP)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        autoencoder.train()
        discriminator.train()

        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_class_loss = 0
        epoch_center_loss = 0
        epoch_perceptual_loss = 0
        epoch_gan_loss = 0
        epoch_total_loss = 0

        kl_weight = min(kl_weight_end,
                        kl_weight_start + (epoch / (num_epochs * 0.6)) * (kl_weight_end - kl_weight_start))
        autoencoder.kl_weight = kl_weight

        print(f"Epoch {epoch + 1}/{num_epochs} - KL Weight: {kl_weight:.6f}")

        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Training")):
            data = data.to(device)
            labels = labels.to(device)
            valid = torch.ones(data.size(0), device=device)
            fake = torch.zeros(data.size(0), device=device)

            optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Use AMP for the forward pass to reduce memory usage
            with torch.amp.autocast("cuda"):
                recon_x, mu, logvar, z = autoencoder(data)

                if epoch < 40:
                    kl_factor = 0.0
                    cls_factor = 0.0
                    center_factor = 0.0
                elif epoch < 80:
                    kl_factor = min(1.0, (epoch - 20) / 20)
                    cls_factor = 0.0
                    center_factor = 0.0
                elif epoch < 160:
                    kl_factor = 1.0
                    cls_factor = min(0.2, (epoch - 40) / 20)
                    center_factor = 0.0
                else:
                    kl_factor = 1.0
                    cls_factor = 1.0
                    center_factor = min(1.0, (epoch - 60) / 20)

                recon_loss = euclidean_distance_loss(recon_x, data)
                perceptual_loss = vgg_loss(recon_x, data)
                kl_loss = autoencoder.kl_divergence(mu, logvar) if kl_factor > 0 else torch.tensor(0.0, device=device)
                class_loss = F.cross_entropy(autoencoder.classify(z), labels) if cls_factor > 0 else torch.tensor(0.0, device=device)
                center_loss = autoencoder.compute_center_loss(z, labels) if center_factor > 0 else torch.tensor(0.0, device=device)

                d_real_loss = gan_criterion(discriminator(data), valid)
                d_fake_loss = gan_criterion(discriminator(recon_x.detach()), fake)
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Train the discriminator
                d_loss.backward()
                d_optimizer.step()

                adv_loss = gan_criterion(discriminator(recon_x), valid)

                if recon_loss.item() > 1e-8:
                    perceptual_scale = min(1.0, recon_loss.item() / (perceptual_loss.item() + 1e-8))
                    kl_scale = min(1.0, recon_loss.item() / (kl_loss.item() + 1e-8)) if kl_loss.item() > 0 else 1.0
                    gan_scale = min(1.0, recon_loss.item() / (adv_loss.item() + 1e-8))
                else:
                    perceptual_scale = 1.0
                    kl_scale = 1.0
                    gan_scale = 1.0

                total_loss = (
                    lambda_recon * recon_loss +
                    lambda_vgg * perceptual_scale * perceptual_loss +
                    kl_weight * kl_scale * kl_factor * kl_loss +
                    lambda_cls * cls_factor * class_loss +
                    lambda_center * center_factor * center_loss +
                    lambda_gan * gan_scale * adv_loss
                )

            # Scale the loss and backpropagate using AMP
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                if epoch >= 60 and center_factor > 0:
                    autoencoder.update_centers(z.detach(), labels, momentum=0.9)

            epoch_recon_loss += recon_loss.item()
            epoch_perceptual_loss += perceptual_loss.item()
            epoch_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0
            epoch_class_loss += class_loss.item() if isinstance(class_loss, torch.Tensor) else 0
            epoch_center_loss += center_loss.item() if isinstance(center_loss, torch.Tensor) else 0
            epoch_total_loss += total_loss.item()
            epoch_gan_loss += adv_loss.item()

            # Optionally remove references to temporary tensors
            del data, labels, recon_x, mu, logvar, z, total_loss

        # Clear cached GPU memory at the end of each epoch
        torch.cuda.empty_cache()

        num_batches = len(train_loader)
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_perceptual_loss = epoch_perceptual_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_class_loss = epoch_class_loss / num_batches
        avg_center_loss = epoch_center_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        avg_gan_loss = epoch_gan_loss / num_batches
        loss_history['recon'].append(avg_recon_loss)
        loss_history['perceptual'].append(avg_perceptual_loss)
        loss_history['kl'].append(avg_kl_loss)
        loss_history['class'].append(avg_class_loss)
        loss_history['center'].append(avg_center_loss)
        loss_history['total'].append(avg_total_loss)
        loss_history['gan'].append(avg_gan_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_total_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, "
            f"Perceptual Loss: {avg_perceptual_loss:.6f}, KL Loss: {avg_kl_loss:.6f}, GAN Loss: {avg_gan_loss:.6f}, "
            f"Class Loss: {avg_class_loss:.6f}, Center Loss: {avg_center_loss:.6f}")

        if loss_history['total'][-1] < best_loss:
            best_loss = loss_history['total'][-1]
            torch.save({
                'autoencoder': autoencoder.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, f"{save_dir}/vae_gan_best.pt")

        if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
            visualize_reconstructions(autoencoder, epoch + 1, save_dir)
            # visualize_latent_space(autoencoder, epoch + 1, save_dir)

    torch.save({
        'autoencoder': autoencoder.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, f"{save_dir}/vae_gan_final.pt")
    print("Training complete.")
    return autoencoder, discriminator, loss_history

def check_and_normalize_latent(autoencoder, data):
    mu, logvar = autoencoder.encode_with_params(data)
    z = autoencoder.reparameterize(mu, logvar)
    mean = z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True)
    z_normalized = (z - mean) / (std + 1e-8)
    return z_normalized, mean, std

def visualize_latent_comparison(autoencoder, diffusion, data_loader, save_path):
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    diffusion.eps_model.eval()
    images, labels = next(iter(data_loader))
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        mu, logvar = autoencoder.encode_with_params(images)
        z = autoencoder.reparameterize(mu, logvar)
        recon = autoencoder.decode(z)
        latent_shape = (images.size(0), autoencoder.latent_dim)
        z_denoised = diffusion.sample(latent_shape, device, c=labels)
        gen = autoencoder.decode(z_denoised)
    n = images.size(0)
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    for i in range(n):
        axes[0, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(gen[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
        axes[2, i].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axes[2, i].axis("off")
    axes[0, 0].set_title("VAE reconstruction（actual latent）", fontsize=10)
    axes[1, 0].set_title("Diffusion（denoised latent）", fontsize=10)
    axes[2, 0].set_title("Original", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    autoencoder.train()
    diffusion.eps_model.train()

def train_conditional_diffusion(autoencoder, unet, train_loader, num_epochs=100, lr=1e-3, visualize_every=10,
                                save_dir="./results", device=None, start_epoch=0):
    print("Starting Class-Conditional Diffusion Model training with improved strategies...")
    os.makedirs(save_dir, exist_ok=True)
    autoencoder.eval()
    diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_history = []
    visualization_loader = train_loader

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0
        for batch_idx, (data, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
            data = data.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                z, batch_mean, batch_std = check_and_normalize_latent(autoencoder, data)
            loss = diffusion.loss(z, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Average Loss: {avg_loss:.6f}")
        scheduler.step()
        if (epoch + 1) % visualize_every == 0 or epoch == start_epoch + num_epochs - 1:
            latent_save_path = os.path.join(save_dir, f"latent_comparison_epoch_{epoch + 1}.png")
            # visualize_latent_comparison(autoencoder, diffusion, visualization_loader, latent_save_path)
            for class_idx in range(min(len(class_names), 2)):
                create_diffusion_animation(autoencoder, diffusion, class_idx=class_idx, num_frames=50,
                                           save_path=f"{save_dir}/diffusion_animation_{class_names[class_idx]}_epoch_{epoch + 1}.gif")
                sample_save_path = f"{save_dir}/sample_class_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                generate_class_samples(autoencoder, diffusion, target_class=class_idx, num_samples=5,
                                       save_path=sample_save_path)
                path_save_path = f"{save_dir}/denoising_path_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                visualize_denoising_steps(autoencoder, diffusion, class_idx=class_idx, save_path=path_save_path)
            torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_epoch_{epoch + 1}.pt")
    torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_final.pt")
    print(f"Saved final diffusion model after {start_epoch + num_epochs} epochs")
    return unet, diffusion, loss_history

def main(checkpoint_path=None, total_epochs=2000):
    print("Starting class-conditional diffusion model for CelebA (Smiling attribute) with improved architecture")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results_dir = "/content/drive/MyDrive/celeba_smiling_conditional_improved"
    os.makedirs(results_dir, exist_ok=True)
    print("Loading CelebA dataset for Smiling attribute...")
    train_dataset = CelebASmiling(root='./data', split="train", transform=transform_train, download=True)
    global class_names
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    autoencoder_path = f"{results_dir}/vae_gan_final.pt"
    diffusion_path = f"{results_dir}/conditional_diffusion_final.pt"
    autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=256, num_classes=2).to(device)
    if os.path.exists(autoencoder_path):
        print(f"Loading existing autoencoder from {autoencoder_path}")
        checkpoint = torch.load(autoencoder_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['autoencoder'], strict=False)
        autoencoder.eval()
    else:
        print("No existing autoencoder found. Training a new one with improved architecture...")
        autoencoder, discriminator, ae_losses = train_autoencoder(
            autoencoder,
            train_loader,
            num_epochs=2000,
            lr=1e-4,
            lambda_cls=0.3,
            lambda_center=0.1,
            lambda_vgg=0.4,
            visualize_every=1,
            save_dir=results_dir
        )
        torch.save(autoencoder.state_dict(), autoencoder_path)
        plt.figure(figsize=(10, 6))
        plt.plot(ae_losses['total'], label='Total Loss')
        plt.plot(ae_losses['recon'], label='Reconstruction Loss')
        plt.plot(ae_losses['kl'], label='KL Loss')
        plt.plot(ae_losses['class'], label='Classification Loss')
        plt.plot(ae_losses['center'], label='Center Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_dir}/autoencoder_losses.png")
        plt.close()
    conditional_unet = ConditionalUNet(
        latent_dim=256,
        hidden_dims=[256, 512, 1024, 512, 256],
        time_emb_dim=256,
        num_classes=2
    ).to(device)

    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            filename = os.path.basename(checkpoint_path)
            epoch_str = filename.split("epoch_")[1].split(".pt")[0]
            start_epoch = int(epoch_str)
            print(f"Continuing training from epoch {start_epoch}")
            conditional_unet.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
        except (IndexError, ValueError) as e:
            print(f"Could not extract epoch number from checkpoint filename: {e}")
            print("Starting from epoch 0")
            start_epoch = 0
    elif os.path.exists(diffusion_path):
        print(f"Loading existing diffusion model from {diffusion_path}")
        conditional_unet.load_state_dict(torch.load(diffusion_path, map_location=device))
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
    else:
        print("No existing diffusion model found. Training a new one with improved architecture...")
        conditional_unet.apply(init_weights)
    remaining_epochs = total_epochs - start_epoch
    if 'diffusion' not in globals():
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, train_loader, num_epochs=remaining_epochs, lr=1e-3,
            visualize_every=50,
            save_dir=results_dir,
            device=device,
            start_epoch=start_epoch
        )
        torch.save(conditional_unet.state_dict(), diffusion_path)
        plt.figure(figsize=(8, 5))
        plt.plot(diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss.png")
        plt.close()
    elif start_epoch > 0:
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, train_loader, num_epochs=remaining_epochs, lr=1e-3,
            visualize_every=50,
            save_dir=results_dir,
            device=device,
            start_epoch=start_epoch
        )
        torch.save(conditional_unet.state_dict(), diffusion_path)
        plt.figure(figsize=(8, 5))
        plt.plot(range(start_epoch + 1, start_epoch + len(diff_losses) + 1), diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss_continued.png")
        plt.close()
    print("Generating sample grid for a subset of classes...")
    grid_path = generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir=results_dir)
    print(f"Sample grid saved to: {grid_path}")
    print("Generating denoising visualizations for a subset of classes...")
    denoising_paths = []
    for class_idx in range(len(class_names)):
        save_path = f"{results_dir}/denoising_path_{class_names[class_idx]}_final.png"
        path = visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=save_path)
        denoising_paths.append(path)
        print(f"Generated visualization for {class_names[class_idx]}")
    print("Creating animations for a subset of classes...")
    for class_idx in range(len(class_names)):
        animation_path = create_diffusion_animation(
            autoencoder, diffusion, class_idx=class_idx,
            num_frames=50, fps=15,
            save_path=f"{results_dir}/diffusion_animation_{class_names[class_idx]}_final.gif"
        )
        print(f"Created animation for {class_names[class_idx]}: {animation_path}")
    print("\nAll visualizations and models complete!")
    print(f"Results directory: {results_dir}")
    print(f"Sample grid: {grid_path}")
    print("Denoising visualizations:")
    for i, path in enumerate(denoising_paths):
        print(f"  - {class_names[i]}: {path}")

if __name__ == "__main__":
    main(total_epochs=10000)
