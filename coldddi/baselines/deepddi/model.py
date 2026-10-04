"""DeepDDI model — Ryu et al. PNAS 2018 (binary adaptation).

Original: concat(SSP_A, SSP_B) -> 9 FC layers (2048 hidden) -> softmax over 86 DDI types.
This adaptation uses a single-logit head for binary DDI classification.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DeepDDIModel(nn.Module):
    """MLP on concat(SSP_A, SSP_B) -> binary logit."""

    def __init__(
        self,
        ssp_dim: int,
        hidden_dim: int = 2048,
        n_layers: int = 9,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        assert n_layers >= 2, "DeepDDI needs at least 2 FC layers"

        layers: List[nn.Module] = []
        in_dim = ssp_dim * 2  # concat of SSP_A and SSP_B
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # binary logit
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ssp_a: torch.Tensor, ssp_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ssp_a, ssp_b: (batch, ssp_dim) float tensors
        Returns:
            logits: (batch,) tensor
        """
        x = torch.cat([ssp_a, ssp_b], dim=-1)
        logits = self.net(x).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, ssp_a: torch.Tensor, ssp_b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(ssp_a, ssp_b))
