"""
GPPyramid Transformer model.

This file contains the core architecture:
1. Pyramidal cross-scale attention.
2. Feature-wise Scale Adaptive Fusion (FSAF).
3. Daily GPP prediction head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_scale_lengths


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal tokens."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class PyramidalCrossScaleAttention(nn.Module):
    """
    Self-attention with a pyramidal structural mask.

    Tokens attend freely within the same scale, while cross-scale attention is
    restricted to parent-child pairs between adjacent temporal scales.
    """

    def __init__(self, d_model, num_heads=4, dropout=0.1, scale_lengths=(365, 183, 92, 46)):
        super().__init__()
        self.scale_lengths = scale_lengths
        self.total_len = sum(scale_lengths)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=self.total_len + 10, dropout=0.0)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("attention_mask", self._build_attention_mask())

    def _build_attention_mask(self):
        n_tokens = self.total_len
        allowed = torch.zeros(n_tokens, n_tokens, dtype=torch.bool)
        offsets = []
        offset = 0
        for length in self.scale_lengths:
            offsets.append(offset)
            offset += length
        for scale_id, length in enumerate(self.scale_lengths):
            offset = offsets[scale_id]
            allowed[offset : offset + length, offset : offset + length] = True
            if scale_id + 1 < len(self.scale_lengths):
                parent_len = self.scale_lengths[scale_id + 1]
                parent_offset = offsets[scale_id + 1]
                fine_idx = torch.arange(length)
                parent_idx = torch.clamp(fine_idx // 2, max=parent_len - 1)
                allowed[offset + fine_idx, parent_offset + parent_idx] = True
            if scale_id - 1 >= 0:
                child_len = self.scale_lengths[scale_id - 1]
                child_offset = offsets[scale_id - 1]
                for t in range(length):
                    for child in (2 * t, 2 * t + 1):
                        if child < child_len:
                            allowed[offset + t, child_offset + child] = True
        return ~allowed

    def forward(self, scale_features, key_padding_mask=None):
        tokens = torch.cat([feature.transpose(1, 2) for feature in scale_features], dim=1)
        tokens = self.pos_encoding(tokens)
        attn_out, _ = self.attention(tokens, tokens, tokens, attn_mask=self.attention_mask, key_padding_mask=key_padding_mask)
        tokens = self.norm(tokens + self.dropout(attn_out))
        outputs = []
        offset = 0
        for length in self.scale_lengths:
            outputs.append(tokens[:, offset : offset + length, :].transpose(1, 2))
            offset += length
        return outputs


class FeatureWiseScaleAdaptiveFusion(nn.Module):
    """
    Feature-wise Scale Adaptive Fusion.

    For each hidden channel and time step, FSAF learns a softmax weight across
    four temporal scales and fuses the aligned multi-scale representations.
    """

    def __init__(self, d_model, target_len=365, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.target_len = target_len
        self.n_scales = 4
        self.upsample_projections = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU())
            for _ in range(3)
        ])
        self.scale_weight_generator = nn.Sequential(
            nn.Conv1d(d_model * self.n_scales, d_model * self.n_scales, kernel_size=1, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * self.n_scales, d_model * self.n_scales, kernel_size=1, groups=d_model),
        )
        self.refine = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout))

    def forward(self, scale_features):
        s1 = scale_features[0]
        s2 = self.upsample_projections[0](F.interpolate(scale_features[1], size=self.target_len, mode="linear", align_corners=False))
        s3 = self.upsample_projections[1](F.interpolate(scale_features[2], size=self.target_len, mode="linear", align_corners=False))
        s4 = self.upsample_projections[2](F.interpolate(scale_features[3], size=self.target_len, mode="linear", align_corners=False))
        stacked = torch.stack([s1, s2, s3, s4], dim=2)
        weight_input = stacked.reshape(stacked.size(0), self.d_model * self.n_scales, stacked.size(-1))
        logits = self.scale_weight_generator(weight_input).reshape(stacked.size(0), self.d_model, self.n_scales, stacked.size(-1))
        weights = F.softmax(logits, dim=2)
        fused = (stacked * weights).sum(dim=2)
        return fused + self.refine(fused)


class FeatureAggregationHead(nn.Module):
    """Prediction head for daily GPP sequences."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden_dim = d_model // 2
        self.aggregation = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
        )
        self.context_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.decoder = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        x = self.aggregation(x)
        gate = self.context_gate(x.mean(dim=-1))
        x = x * gate.unsqueeze(-1)
        return self.decoder(x).squeeze(1)


class GPPyramidTransformer(nn.Module):
    """
    Pyramidal Transformer for daily GPP estimation.

    Inputs:
        hf: daily high-frequency drivers, shaped as (B, C, 365)
        lai_s1, lai_s2, lai_s3, lai_s4: LAI sequences aligned to each scale
        pft_id: plant functional type index
        input_mask: missing-feature mask at daily resolution
    """

    def __init__(self, hf_in, n_pft, d_model=64, num_heads=2, dropout=0.3, target_len=365, n_pyramid_layers=2):
        super().__init__()
        self.d_model = d_model
        self.target_len = target_len
        self.scale_lengths = compute_scale_lengths(target_len)
        self.driver_embedding = nn.Conv1d(hf_in, d_model, kernel_size=1)
        self.pft_embedding = nn.Embedding(n_pft, d_model)
        self.downsample_layers = nn.ModuleList([nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) for _ in range(3)])
        self.downsample_norms = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(3)])
        self.lai_projections = nn.ModuleList([nn.Conv1d(d_model + 1, d_model, kernel_size=1) for _ in range(4)])
        self.cross_scale_layers = nn.ModuleList([
            PyramidalCrossScaleAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, scale_lengths=self.scale_lengths)
            for _ in range(n_pyramid_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
                for _ in range(4)
            ])
            for _ in range(n_pyramid_layers)
        ])
        self.fusion = FeatureWiseScaleAdaptiveFusion(d_model=d_model, target_len=target_len, dropout=dropout)
        self.head = FeatureAggregationHead(d_model=d_model, dropout=dropout)

    def _downsample_mask(self, input_mask):
        scale_masks = [input_mask]
        mask = input_mask.float().unsqueeze(1)
        for _ in range(3):
            mask = F.max_pool1d(mask, kernel_size=2, stride=2, padding=0, ceil_mode=True)
            scale_masks.append(mask.squeeze(1).bool())
        return scale_masks

    def forward(self, hf, lai_s1, lai_s2, lai_s3, lai_s4, pft_id, input_mask):
        x = self.driver_embedding(hf)
        x = x + self.pft_embedding(pft_id).unsqueeze(-1)
        scale_features = []
        current = x
        for scale_id in range(4):
            scale_features.append(current)
            if scale_id < 3:
                current = F.gelu(self.downsample_norms[scale_id](self.downsample_layers[scale_id](current)))
        lai_inputs = [lai_s1, lai_s2, lai_s3, lai_s4]
        for scale_id in range(4):
            scale_features[scale_id] = F.gelu(self.lai_projections[scale_id](torch.cat([scale_features[scale_id], lai_inputs[scale_id]], dim=1)))
        key_padding_mask = torch.cat(self._downsample_mask(input_mask), dim=1)
        for layer_id, attention_layer in enumerate(self.cross_scale_layers):
            scale_features = attention_layer(scale_features, key_padding_mask=key_padding_mask)
            updated_features = []
            for scale_id, feature in enumerate(scale_features):
                feature_t = feature.transpose(1, 2)
                feature_t = feature_t + self.ffn_layers[layer_id][scale_id](feature_t)
                updated_features.append(feature_t.transpose(1, 2))
            scale_features = updated_features
        fused = self.fusion(scale_features)
        return self.head(fused)
