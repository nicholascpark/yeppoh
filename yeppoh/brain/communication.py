"""Inter-agent communication channel.

★ THIS IS WHERE EMERGENT LANGUAGE HAPPENS.

Each agent broadcasts a learned vector each step. Nearby agents
receive these messages and incorporate them into their observations.
Over training, agents develop a functional communication protocol.

Supports:
- Broadcast (all neighbors hear the same message)
- Targeted (attention-weighted messages to specific neighbors)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommChannel(nn.Module):
    """Differentiable communication channel between agents.

    Each agent encodes a message from its features and decodes
    incoming messages from neighbors into a context vector.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        message_dim: int = 16,
        n_heads: int = 2,
        max_neighbors: int = 4,
    ):
        super().__init__()
        self.message_dim = message_dim
        self.n_heads = n_heads
        self.max_neighbors = max_neighbors

        # Encode features → outgoing message
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, message_dim),
        )

        # Decode incoming messages → context vector
        # Multi-head attention over neighbor messages
        self.query_proj = nn.Linear(feature_dim, message_dim * n_heads)
        self.key_proj = nn.Linear(message_dim, message_dim * n_heads)
        self.value_proj = nn.Linear(message_dim, message_dim * n_heads)
        self.output_proj = nn.Linear(message_dim * n_heads, feature_dim)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode agent features into an outgoing message.

        Args:
            features: (batch, feature_dim)

        Returns:
            message: (batch, message_dim) — broadcast to neighbors
        """
        return self.encoder(features)

    def decode(
        self,
        features: torch.Tensor,
        neighbor_messages: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode incoming neighbor messages into a context vector.

        Uses multi-head attention: the agent's features generate queries,
        neighbor messages generate keys and values.

        Args:
            features: (batch, feature_dim) — this agent's features
            neighbor_messages: (batch, max_neighbors, message_dim)
            neighbor_mask: (batch, max_neighbors) bool — True = valid neighbor

        Returns:
            context: (batch, feature_dim) — integrated neighbor information
        """
        batch, n_nbr, mdim = neighbor_messages.shape

        # Project
        Q = self.query_proj(features).reshape(batch, self.n_heads, self.message_dim)
        K = self.key_proj(neighbor_messages.reshape(-1, mdim))
        K = K.reshape(batch, n_nbr, self.n_heads, self.message_dim).permute(0, 2, 1, 3)
        V = self.value_proj(neighbor_messages.reshape(-1, mdim))
        V = V.reshape(batch, n_nbr, self.n_heads, self.message_dim).permute(0, 2, 1, 3)

        # Attention scores
        scale = self.message_dim ** 0.5
        scores = (Q.unsqueeze(2) @ K.transpose(-2, -1)).squeeze(2) / scale  # (batch, heads, n_nbr)

        if neighbor_mask is not None:
            mask = neighbor_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=-1)

        # Weighted sum of values
        context = (attn.unsqueeze(-1) * V).sum(dim=2)  # (batch, heads, mdim)
        context = context.reshape(batch, self.n_heads * self.message_dim)

        return self.output_proj(context)
