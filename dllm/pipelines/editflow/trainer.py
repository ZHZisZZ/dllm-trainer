from typing import Any, Dict, Union, List, Tuple, Optional
from dataclasses import dataclass
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from dllm.utils.schedulers import BaseScheduler, LinearScheduler


BLANK = -1

def align_with_blanks(x0: List[int], x1: List[int], sub_cost: int = 1, gap_cost: int = 1) -> Dict:
    """
    Needleman–Wunsch global alignment of two integer sequences with:
        match cost = 0, substitution cost = sub_cost, gap cost = gap_cost.
    Returns aligned sequences (z0, z1) of equal length containing BLANK = ε where gaps occur.
    """
    n, m = len(x0), len(x1)
    # DP tables
    dp = [[0]*(m+1) for _ in range(n+1)]
    ptr = [[None]*(m+1) for _ in range(n+1)]  # 'diag', 'up', 'left'

    for i in range(1, n+1):
        dp[i][0] = i * gap_cost
        ptr[i][0] = 'up'
    for j in range(1, m+1):
        dp[0][j] = j * gap_cost
        ptr[0][j] = 'left'

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost_diag = dp[i-1][j-1] + (0 if x0[i-1] == x1[j-1] else sub_cost)
            cost_up   = dp[i-1][j] + gap_cost
            cost_left = dp[i][j-1] + gap_cost
            best = min(cost_diag, cost_up, cost_left)
            dp[i][j] = best
            if best == cost_diag:
                ptr[i][j] = 'diag'
            elif best == cost_up:
                ptr[i][j] = 'up'
            else:
                ptr[i][j] = 'left'

    # traceback
    z0, z1 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        p = ptr[i][j]
        if p == 'diag':
            z0.append(x0[i-1])
            z1.append(x1[j-1])
            i -= 1; j -= 1
        elif p == 'up':
            z0.append(x0[i-1])
            z1.append(BLANK)
            i -= 1
        else:  # 'left'
            z0.append(BLANK)
            z1.append(x1[j-1])
            j -= 1
    z0.reverse(); z1.reverse()
    # return Alignment(z0=z0, z1=z1)
    return 


def strip_blanks(z: List[int]) -> List[int]:
    # IMPORTANT: do NOT strip BOS; we only remove BLANKs
    return [t for t in z if t != BLANK]


# -----------------------------
# κ(t) schedule
# -----------------------------

def kappa(t: torch.Tensor) -> torch.Tensor:
    """
    Monotone schedule κ: [0,1]->[0,1].
    We use smooth cosine (slower start, faster end): κ(t) = (1 - cos(π t)) / 2
    """
    return 0.5 * (1 - torch.cos(math.pi * t))


def kappa_dot(t: torch.Tensor) -> torch.Tensor:
    """Derivative of κ(t) used for weighting: κ˙(t) = (π/2) sin(π t)."""
    return 0.5 * math.pi * torch.sin(math.pi * t)


# -----------------------------
# Collation helpers (ragged -> padded tensors)
# -----------------------------

def pad_1d(batch_lists: List[List[int]], pad_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of variable-length integer lists into a tensor [B, Lmax] plus mask [B, Lmax].
    """
    B = len(batch_lists)
    Lmax = max((len(x) for x in batch_lists), default=0)
    out = torch.full((B, Lmax), pad_val, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)
    for b, x in enumerate(batch_lists):
        if len(x) == 0:
            continue
        out[b, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[b, :len(x)] = True
    return out, mask


# -----------------------------
# Loss utilities
# -----------------------------

# def mask_out_current_token(sub_logits: torch.Tensor, x_tok: torch.Tensor, x_mask: torch.Tensor, V: int) -> torch.Tensor:
#     """
#     Zero probability (−inf logit) for substituting a token with itself, at valid positions.
#     sub_logits : [B,L,V]
#     x_tok      : [B,L]
#     x_mask     : [B,L]
#     """
#     B, L, Vv = sub_logits.shape
#     assert Vv == V
#     idx = torch.clamp(x_tok, min=0).unsqueeze(-1)  # [B,L,1]
#     big_neg = -1e9
#     sub_logits = sub_logits.scatter(2, idx, big_neg)
#     return sub_logits


@dataclass
class Edit:
    kind: str            # "SUB" | "DEL" | "INS"
    pos_or_gap: int      # position (for SUB/DEL) or token-row idx for INS (incl. BOS row 0)
    token: Optional[int] # token for SUB/INS, else None


def build_remaining_edits(zt: List[int], z1: List[int]) -> List[Edit]:
    edits: List[Edit] = []
    def count_nonblank_prefix(z: List[int], j: int) -> int:
        c = 0
        for k in range(j):
            if z[k] != BLANK:
                c += 1
        return c

    for j, (a, b) in enumerate(zip(zt, z1)):
        if a == b:
            continue
        nb = count_nonblank_prefix(zt, j)  # counts BOS as 1, first content token will be nb=1 before its column

        if a == BLANK and b != BLANK:
            # INSERT after row (nb-1): BOS insert => nb=1 -> gap=0; general case works too
            gap = max(nb - 1, 0)
            edits.append(Edit("INS", gap, b))

        elif a != BLANK and b == BLANK:
            # DELETE token at row nb (first content token => nb=1, allowed; BOS is never BLANK so nb>=1)
            pos = nb
            # if pos > 0:   # forbid BOS (row 0)
            edits.append(Edit("DEL", pos, None))

        else:  # a != BLANK, b != BLANK, a != b
            # SUB token at row nb
            pos = nb
            # if pos > 0:   # forbid BOS (row 0)
            edits.append(Edit("SUB", pos, b))
    return edits



class EditFlowTrainer(transformers.Trainer):

    def __init__(
        self,
        *args,
        # scheduler: BaseScheduler = LinearScheduler(),
        **kawrgs,
    ):
        # self.scheduler = scheduler
        return super().__init__(*args, **kawrgs)

    def compute_loss(
        self,
        model: Union[transformers.PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs = False,
        **kwargs,
    ):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
        # input_ids, labels = inputs["input_ids"], inputs["labels"]
        # input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        # breakpoint()
        # pass
        device = self.model.device
        aligns = [align_with_blanks(x0, x1) for x0, x1 in zip(inputs["x0_ids"], inputs["x1_ids"])]
        z0_list = [a["z0"] for a in aligns]
        z1_list = [a["z1"] for a in aligns]
        assert all(len(z0)==len(z1) for z0, z1 in zip(z0_list, z1_list))

        # 2) Sample time and z_t via Bernoulli(κ(t))
        t = torch.rand(len(z0_list), 1, device=device)  # [B,1]
        k = kappa(t)                                    # [B,1]
        zt_list: List[List[int]] = []
        for b, (z0, z1) in enumerate(zip(z0_list, z1_list)):
            M = len(z0)
            kb = k[b].item()
            zt = []
            for j in range(M):
                take_target = (random.random() < kb)
                # BOS is the same token in z0/z1; mixing preserves it automatically
                zt.append(z1[j] if take_target else z0[j])
            zt_list.append(zt)

        # 3) Strip blanks -> x_t and build remaining edits (BOS remains)
        xt_list = [strip_blanks(zt) for zt in zt_list]
        edits_list: List[List[Edit]] = [build_remaining_edits(zt, z1) for zt, z1 in zip(zt_list, z1_list)]

        # 4) Collate x_t for the model
        x_tok, x_mask = pad_1d(xt_list, pad_val=self.processing_class.pad_token)  # [B,Lmax], [B,Lmax]
        x_tok = x_tok.to(device)
        x_mask = x_mask.to(device)

        # 5) Forward
        out = model(x_tok, x_mask, t.to(device))  # dict of tensors
        # Token/Gap distributions
        # sub_logits = mask_out_current_token(out["sub_logits"], x_tok, x_mask, V) # NOTE: i think this is optional; but better keep it
        Q_sub = F.log_softmax(out["sub_logits"], dim=-1)  # log-probs for stability
        Q_ins = F.log_softmax(out["ins_logits"], dim=-1)

        # 6) Build loss
        # Per Eq. 23: survival (sum of intensities) UNWEIGHTED,
        # positive selected-edit term weighted by w = κ˙/(1-κ).
        w = (kappa_dot(t) / (1 - k + 1e-6)).squeeze(1)  # [B]
        loss_surv = torch.tensor(0.0, device=device)
        loss_pos = torch.tensor(0.0, device=device)
        n_pos = 0

        # # Survival: sum of intensities at current x_t
        # Lambda_all = (
        #     out["sub_intensity"].sum(dim=1)  # [B]
        #     + out["del_intensity"].sum(dim=1)
        #     + out["ins_intensity"].sum(dim=1)
        # )  # [B]
        # loss_surv = Lambda_all.mean()  # UNWEIGHTED survival term
        # Build masks in the loss
        # cur_len = x_mask.sum(dim=1)                    # includes BOS
        # pos_mask = x_mask.clone()                      # valid token rows
        # pos_mask[:, 0] = False                         # exclude BOS row for sub/del

        # Survival term with masks applied here (no need to mask in forward)
        Lambda_all = (
                (out["sub_intensity"] * x_mask.float()).sum(dim=1)
            + (out["del_intensity"] * x_mask.float()).sum(dim=1)
            + (out["ins_intensity"] * x_mask.float()).sum(dim=1)  # insert allowed at BOS
        )
        loss_surv = Lambda_all.mean()

        # Positive terms: -w * log(rate) for each remaining edit
        for b, edits in enumerate(edits_list):
            if len(edits) == 0:
                continue
            n_pos += len(edits)
            cur_len = int(x_mask[b].sum().item())  # includes BOS
            for e in edits:
                if e.kind == "SUB":
                    pos = e.pos_or_gap
                    tok = e.token
                    if pos >= cur_len or pos == 0:  # skip BOS row
                        continue
                    log_q = Q_sub[b, pos, tok]                    # log Q_sub
                    lam = out["sub_intensity"][b, pos]           # λ_sub
                    loss_pos = loss_pos - w[b] * (log_q + torch.log(lam + 1e-12))
                elif e.kind == "DEL":
                    pos = e.pos_or_gap
                    if pos >= cur_len or pos == 0:  # skip BOS row
                        continue
                    lam = out["del_intensity"][b, pos]           # λ_del
                    loss_pos = loss_pos - w[b] * torch.log(lam + 1e-12)
                else:  # "INS"
                    gap = e.pos_or_gap  # now a token row index incl. BOS
                    tok = e.token
                    gap_max = cur_len  # valid gaps are rows [0..cur_len-1]
                    if gap >= gap_max:
                        continue
                    log_q = Q_ins[b, gap, tok]                   # log Q_ins
                    lam = out["ins_intensity"][b, gap]           # λ_ins
                    loss_pos = loss_pos - w[b] * (log_q + torch.log(lam + 1e-12))

        if n_pos > 0:
            loss_pos = loss_pos / n_pos

        loss = loss_surv + loss_pos

        return (loss, out) if return_outputs else loss


if __name__ == "__main__":
    pass
