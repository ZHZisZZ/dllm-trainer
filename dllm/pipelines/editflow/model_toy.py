from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise SystemExit("This script requires PyTorch. Please install with `pip install torch`.\n"
                     f"Original import error: {e}")


class EditFlowToy(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128):
        super().__init__()
        self.V = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )
        # Heads for positions (substitute, delete)
        self.sub_intensity = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.sub_logits    = nn.Linear(d_model, vocab_size)
        self.del_intensity = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        # Heads for insertion now use token rows directly (incl. BOS)
        self.ins_intensity = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.ins_logits    = nn.Linear(d_model, vocab_size)
        # (boundary/gap mlp no longer needed; kept out for minimal change)

    def forward(self, x_tok: torch.Tensor, x_mask: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_tok : [B, L] Long (PAD where mask is False). BOS is present at index 0 where mask is True.
        x_mask: [B, L] Bool
        t     : [B, 1] Float in [0,1]
        Returns:
            - sub_intensity: [B, L] (0 on pads; also 0 at BOS)
            - sub_logits:    [B, L, V] (-inf on pads; -inf at BOS)
            - del_intensity: [B, L] (0 on pads; also 0 at BOS)
            - ins_intensity: [B, L]          # insert AFTER each row, incl. BOS row
            - ins_logits:    [B, L, V]
        """
        B, L = x_tok.shape
        big_neg = -1e9

        # Embedding with masking (negative special tokens clamp to 0; masked later)
        tok_emb = self.embed(torch.clamp(x_tok, min=0))  # [B, L, d]
        t_emb = self.time_mlp(t).unsqueeze(1).expand(B, L, -1)  # [B, L, d]
        h = tok_emb + t_emb

        # Transformer encoder (mask pads as True in src_key_padding_mask)
        h = self.encoder(h, src_key_padding_mask=~x_mask)  # [B, L, d]

        # Position heads
        sub_int = self.sub_intensity(h).squeeze(-1)     # [B, L]
        sub_log = self.sub_logits(h)                    # [B, L, V]
        del_int = self.del_intensity(h).squeeze(-1)     # [B, L]

        # Insertion heads now on token rows (incl. BOS)
        ins_int = self.ins_intensity(h).squeeze(-1)     # [B, L]
        ins_log = self.ins_logits(h)                    # [B, L, V]

        # # Mask pads on all heads
        # sub_int = sub_int * x_mask.float()
        # del_int = del_int * x_mask.float()
        # ins_int = ins_int * x_mask.float()
        # sub_log = sub_log.masked_fill(~x_mask.unsqueeze(-1), big_neg)
        # ins_log = ins_log.masked_fill(~x_mask.unsqueeze(-1), big_neg)

        # Forbid delete/substitute at BOS row (index 0 when mask is True)
        # if L > 0:
        #     bos_mask = x_mask[:, 0:1]  # True where BOS row exists
        #     sub_int[:, 0] = sub_int[:, 0] * (~bos_mask.squeeze(1)).float()
        #     del_int[:, 0] = del_int[:, 0] * (~bos_mask.squeeze(1)).float()
        #     sub_log[:, 0, :] = sub_log[:, 0, :].masked_fill(bos_mask, big_neg)

        return dict(
            sub_intensity=sub_int,    # [B,L]
            sub_logits=sub_log,       # [B,L,V]
            del_intensity=del_int,    # [B,L]
            ins_intensity=ins_int,    # [B,L]
            ins_logits=ins_log,       # [B,L,V]
        )
