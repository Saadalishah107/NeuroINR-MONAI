"""
src/models/vit.py

A compact Vision Transformer, trained from scratch on the same MRI slices
as the CNN baseline. This is deliberately NOT a pretrained ImageNet ViT
(those expect 3-channel natural images and RGB-statistics normalisation
that don't transfer meaningfully to single-channel MRI without a much
larger fine-tuning dataset than we have here). Presented as an extension
/ comparison point against the CNN baseline, not as the primary model --
see README "Core vs Extensions".
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)       # (B, n_patches, embed_dim)
        return x


class TinyViT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1,
                 embed_dim=128, depth=4, n_heads=4, mlp_ratio=2.0,
                 n_classes=2, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
