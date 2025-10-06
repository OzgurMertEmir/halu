from __future__ import annotations
from typing import Set
import torch
from .types import ForwardPack

class HFRunner:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        store_device: str = "cpu",
        store_dtype: torch.dtype = torch.float16,
        include_embeddings: bool = False,
        keep_logits_full: bool = False,
    ):
        self.model = model
        self.tok = tokenizer
        self.store_device = store_device

        if store_dtype is None:
            store_dtype = torch.float16
        self.store_dtype = store_dtype

        bf16_ok = (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
                  or (torch.backends.mps.is_available()) \
                  or (torch.backends.mkldnn.is_available() and torch.cpu.is_bf16_supported())
        self.store_dtype_states = torch.bfloat16 if bf16_ok else torch.float16
        self.store_dtype_logits = torch.float32
        self.include_embeddings = include_embeddings
        self.keep_logits_full = keep_logits_full

    def _model_device(self):
        try: return next(self.model.parameters()).device
        except StopIteration: return torch.device("cpu")

    def _place(self, t, dtype=None):
        dev = (self._model_device() if self.store_device == "gpu" else torch.device("cpu"))
        return t.to(device=dev, dtype=(dtype or self.store_dtype), non_blocking=True)

    @torch.inference_mode()
    def prompt_response_forward(self, prompt: str, response: str, need: Set[str]) -> ForwardPack:
        model, tok = self.model, self.tok
        model.eval()
        text = prompt + response
        enc = tok(text, return_tensors="pt").to(model.device)
        prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
        total_len  = enc.input_ids.shape[1]
        resp_len   = total_len - prompt_len
        assert resp_len > 0, "Empty response region."

        normalized = set()
        for k in need:
            if k in {"attn", "attn_layers"}:
                normalized.add("attn_layers")
            elif k in {"hidden", "hidden_layers"}:
                normalized.add("hidden_layers")
            else:
                normalized.add(k)
        need = normalized

        want_attn_layers   = ("attn_layers" in need)
        want_hidden_layers = ("hidden_layers" in need)
        want_logits        = ("logits" in need) or ("logits_full" in need)

        out = model(
            **enc,
            output_attentions=want_attn_layers,
            output_hidden_states=want_hidden_layers,
            use_cache=False,
            return_dict=True,
        )

        logits_full = None
        logits_resp = None
        if want_logits:
            lf = out.logits[0]
            if self.keep_logits_full or ("logits_full" in need):
                logits_full = self._place(lf, dtype=self.store_dtype_logits)
            if prompt_len == 0:
                logits_resp = self._place(lf[: total_len - 1][-resp_len:], dtype=self.store_dtype_logits)
            else:
                logits_resp = self._place(lf[prompt_len - 1 : total_len - 1], dtype=self.store_dtype_logits)

        attn_layers = None
        if want_attn_layers:
            assert out.attentions is not None
            attn_layers = [ self._place(A[0], dtype=self.store_dtype_states) for A in out.attentions ]

        hidden_layers = None
        hidden_embed = None
        if want_hidden_layers:
            assert out.hidden_states is not None
            hs = out.hidden_states
            if self.include_embeddings:
                hidden_embed = self._place(hs[0][0], dtype=self.store_dtype_states)
            hidden_layers = [ self._place(hs[i][0], dtype=self.store_dtype_states) for i in range(1, len(hs)) ]

        return ForwardPack(
            prompt=prompt,
            response=response,
            prompt_len=int(prompt_len),
            total_len=int(total_len),
            resp_len=int(resp_len),
            logits_resp=logits_resp,
            logits_full=logits_full,
            attn_layers=attn_layers,
            hidden_layers=hidden_layers,
            hidden_embed=hidden_embed,
        )
