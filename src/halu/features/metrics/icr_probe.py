from __future__ import annotations
from typing import Iterable, Optional, Dict
import torch
from core.types import ForwardPack, MCQExample
from core.utils import _safe_cos, pool_icr_features, EPS

class ICRProbeMetric:
    """ICR-Probe (MCQ): track hidden-state dynamics over response tokens."""
    needs = ["hidden_layers", "attn_layers"]  # attention optional

    def __init__(
        self,
        layers: Optional[Iterable[int]] = None,
        fixed_layer: int = -1,
        use_attention: bool = True,
    ):
        self.layers = tuple(layers) if layers is not None else (-4, -3, -2, -1)
        self.fixed_layer = fixed_layer
        self.use_attention = use_attention

    @torch.no_grad()
    def compute(self, pack: ForwardPack, ex: MCQExample | None = None) -> Dict[str, float]:
        assert pack.hidden_layers is not None and len(pack.hidden_layers) >= 2, \
            "ICRProbeMetric needs per-layer hidden states (len >= 2)."

        T_resp = int(pack.resp_len)
        if T_resp <= 0:
            return {"icrp_resp_len": 0}

        start = int(pack.prompt_len)
        sl = slice(start, start + T_resp)

        L_all = len(pack.hidden_layers)
        def _norm(li: int) -> int: return li if li >= 0 else (L_all + li)
        layer_ids = [i for i in map(_norm, self.layers) if 0 <= i < L_all]
        if len(layer_ids) < 2:
            layer_ids = [max(0, L_all-2), L_all-1]

        Hs = [pack.hidden_layers[i][sl, :].to(torch.float32) for i in layer_ids]  # [T,d] * L'
        layer_deltas = [Hs[j] - Hs[j-1] for j in range(1, len(Hs))]              # [T,d]
        if not layer_deltas:
            return {"icrp_resp_len": T_resp}

        dL = torch.stack(layer_deltas, dim=0)       # [L', T, d]
        dL_norm = dL.norm(dim=-1)                   # [L', T]
        delta_per_tok = dL_norm.mean(dim=0)         # [T]

        layer_delta_mean = float(delta_per_tok.mean().item())
        layer_delta_std  = float(delta_per_tok.std(unbiased=False).item())
        layer_delta_cv   = float(layer_delta_std / max(layer_delta_mean, EPS))
        layer_delta_p95  = float(delta_per_tok.quantile(0.95).item())

        interlayer_cos_mean = float('nan')
        interlayer_cos_tok = None
        if dL.shape[0] >= 2:
            interlayer_cos_tok = _safe_cos(dL[1:], dL[:-1]).mean(dim=0)  # [T]
            interlayer_cos_mean = float(interlayer_cos_tok.mean().item())

        fix_idx = min(max(_norm(self.fixed_layer), 0), L_all - 1)
        Hfix = pack.hidden_layers[fix_idx][sl, :].to(torch.float32)
        time_delta_mean = time_delta_cv = time_delta_p95 = float('nan')
        temporal_cos_mean = float('nan')
        dT_norm = None
        temporal_cos_tok = None
        if Hfix.shape[0] >= 2:
            dT = Hfix[1:] - Hfix[:-1]  # [T-1,d]
            dT_norm = dT.norm(dim=-1)
            time_delta_mean = float(dT_norm.mean().item())
            time_delta_std  = float(dT_norm.std(unbiased=False).item())
            time_delta_cv   = float(time_delta_std / max(time_delta_mean, EPS))
            time_delta_p95  = float(dT_norm.quantile(0.95).item())
            if dT.shape[0] >= 2:
                temporal_cos_tok = _safe_cos(dT[1:], dT[:-1])  # [T-2]
                temporal_cos_mean = float(temporal_cos_tok.mean().item())

        pm_tok = None
        """
        if self.use_attention and getattr(pack, "attn_layers", None) is not None and pack.prompt_len > 0:
            A = pack.attn_layers[fix_idx].to(torch.float32)   # [H,T,T]
            A_mean = A.mean(dim=0)                            # [T,T]
            A_resp_prompt = A_mean[sl, :pack.prompt_len]      # [T, prompt_len]
            if A_resp_prompt.numel() > 0:
                pm_tok = A_resp_prompt.sum(dim=-1).clamp(0.0, 1.0)"""


        if self.use_attention and getattr(pack, "attn_layers", None) and pack.prompt_len > 0:
            # If only one attention layer was stored (batched path), use that.
            # Otherwise, align with fixed_layer like before.
            if len(pack.attn_layers) == 1:
                A = pack.attn_layers[0].to(torch.float32)  # [H,T,T]
            else:
                attn_idx = min(max(_norm(self.fixed_layer), 0), len(pack.attn_layers) - 1)
                A = pack.attn_layers[attn_idx].to(torch.float32)  # [H,T,T]

            A_mean = A.mean(dim=0)                       # [T,T]
            A_resp_prompt = A_mean[sl, :pack.prompt_len] # [T, prompt_len]
            if A_resp_prompt.numel() > 0:
                pm_tok = A_resp_prompt.sum(dim=-1).clamp(0.0, 1.0)

        pooled = pool_icr_features(
            delta_per_tok=delta_per_tok,
            dL_norm=dL_norm,
            dT_norm=dT_norm,
            interlayer_cos_tok=interlayer_cos_tok,
            temporal_cos=temporal_cos_tok,
            prompt_mass_tok=pm_tok,
        )

        return {
            "icrp_layer_delta_mean": layer_delta_mean,
            "icrp_layer_delta_cv":   layer_delta_cv,
            "icrp_layer_delta_p95":  layer_delta_p95,
            "icrp_time_delta_mean":  time_delta_mean,
            "icrp_time_delta_cv":    time_delta_cv,
            "icrp_time_delta_p95":   time_delta_p95,
            "icrp_interlayer_cos_mean": interlayer_cos_mean,
            "icrp_temporal_cos_mean":   temporal_cos_mean,
            "icrp_prompt_mass_mean":    float(pm_tok.mean().item()) if pm_tok is not None else float('nan'),
            "icrp_pool_attn_cons":      pooled.get("attn_delta_consistency", float('nan')),
            "icrp_resp_len":            T_resp,
            "icrp_pool_vec":            pooled.get("pooled_vector").tolist() if pooled.get("pooled_vector") is not None else [],
            "icrp_reliable": int(T_resp >= 5),
        }
