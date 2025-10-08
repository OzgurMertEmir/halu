# src/halu/features/metrics/llmcheck.py
from __future__ import annotations
from typing import Dict, Optional, Union
import re, torch
from halu.core.types import ForwardPack, MCQExample
from halu.core.utils import EPS
from halu.core import utils, prompts

class LLMCheckMetric:
    """
    PMI-like features on a chosen span of the option.

    target ∈ {"letter","content"}:
      - "letter"  : score the first response token (choice letter)
      - "content" : score tokens after the 'A) ' / 'B)' prefix

    span_mode ∈ {"first","all", int}:
      - "first" -> just one token from the chosen target
      - "all"   -> use every available token for the chosen target
      - int N   -> use up to N tokens for the chosen target
    """
    needs = ["logits"]

    def __init__(
        self,
        runner=None,
        *,
        target: str = "content",
        span_mode: Union[str, int] = "first",
    ):
        assert target in {"letter","content"}
        if isinstance(span_mode, str):
            assert span_mode in {"first","all"}
        else:
            assert isinstance(span_mode, int) and span_mode > 0

        self.runner = runner
        self.target = target
        self.span_mode = span_mode

        # Prefix names so you can run both views (letter/content) side-by-side
        self.prefix = f"llmc_{self.target}_"

    def _content_offset_tokens(self, tok, response: str, label: Optional[str]) -> int:
        """
        Return the token count of the letter prefix (e.g., 'A) ') in `response`.
        Uses add_special_tokens=False to measure raw text length.
        """
        if label:
            # Normalize and build explicit prefix like "A) "
            prefix = f"{label.strip().upper()}) "
        else:
            # Fallback: parse from the response text
            m = re.match(r"\s*([A-Za-z])\s*\)\s*", response or "")
            prefix = m.group(0) if m else ""
        ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids
        if ids.ndim == 2:
            return int(ids.shape[1])
        return int(ids.shape[0])

    @torch.no_grad()
    def compute(self, pack: ForwardPack, ex: MCQExample, option_label: Optional[str] = None,
           cached_q_prompt: Optional[str] = None,
           cached_q_prompt_len: Optional[int] = None) -> Dict[str, float]:
        model = self.runner.model
        tok   = self.runner.tok
        model.eval()

        prompt_len = int(pack.prompt_len)
        resp_T     = int(pack.resp_len)

        # Early guard
        if resp_T <= 0 or pack.logits_resp is None or pack.logits_resp.shape[0] == 0:
            return {
                f"{self.prefix}ce_cond": float("nan"),
                f"{self.prefix}ce_uncond": float("nan"),
                f"{self.prefix}ce_logpmi": float("nan"),
                f"{self.prefix}logit_entropy_mean": float("nan"),
                f"{self.prefix}span_len": 0,
                "llmc_prompt_len": prompt_len,
                "llmc_total_len": int(pack.total_len),
                "llmc_resp_len": resp_T,
            }

        # --- choose the span inside the response ---
        if self.target == "letter":
            t0 = 0
        else:
            t0 = min(self._content_offset_tokens(tok, pack.response, option_label), max(resp_T - 1, 0))
            #t0 = min(self._content_offset_tokens(tok, pack.response), max(resp_T - 1, 0))

        if self.span_mode == "first":
            span_req = 1
        elif self.span_mode == "all":
            span_req = resp_T - t0
        else:
            span_req = int(self.span_mode)
        span_len = max(1, min(span_req, resp_T - t0))

        # ---------- CONDITIONAL ----------
        # LETTER: keep your existing cond text (prompt+options).
        # CONTENT: **question-only** conditional (NO options list) to avoid leakage.
        if self.target == "content":
            if cached_q_prompt is not None and cached_q_prompt_len is not None:
                # Fast path: reuse cached question-only prompt
                text_cond = cached_q_prompt + pack.response
                enc_c = tok(text_cond, return_tensors="pt", add_special_tokens=False).to(model.device)
                prompt_len_c = int(cached_q_prompt_len)
            else:
                # Fallback (current behavior)
                system_txt = "You are a helpful, factual assistant."
                user_body  = f"Question: {ex.question}\nAnswer:"
                if hasattr(tok, "apply_chat_template"):
                    msgs = [{"role":"system","content":system_txt},{"role":"user","content":user_body}]
                    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    text_cond = base + pack.response
                    prompt_len_c = int(tok(base, return_tensors="pt", add_special_tokens=False).input_ids.shape[1])
                else:
                    base = system_txt + "\n\n" + user_body
                    text_cond = base + " " + pack.response
                    prompt_len_c = int(tok(base, return_tensors="pt", add_special_tokens=False).input_ids.shape[1])

                enc_c = tok(text_cond, return_tensors="pt", add_special_tokens=False).to(
                        model.device)
            # Re-run a tiny forward to get logits on the aligned text
            out_c = model(**enc_c, use_cache=False, return_dict=True)
            logits_c_full = out_c.logits[0]  # [T,V]
            # Align to the content span
            start_abs = prompt_len_c + t0
            end_abs   = min(start_abs + span_len, enc_c.input_ids.shape[1])
            logits_c = logits_c_full[start_abs-1:end_abs-1, :]
            targets_c = enc_c.input_ids[0, start_abs:end_abs]
        else:
            # letter uses existing pack (original prompt+options)
            text_cond = pack.prompt + pack.response
            enc_c = tok(text_cond, return_tensors="pt", add_special_tokens=False).to(model.device)
            logits_c = pack.logits_resp[t0:t0+span_len].to(torch.float32)
            start_abs = prompt_len + t0
            end_abs   = min(start_abs + span_len, enc_c.input_ids.shape[1])
            targets_c = enc_c.input_ids[0, start_abs:end_abs]

        # defensive trim
        if logits_c.shape[0] != targets_c.shape[0]:
            L = min(logits_c.shape[0], targets_c.shape[0])
            logits_c = logits_c[:L]
            targets_c = targets_c[:L]
            span_len = int(L)

        ce_cond = utils._mean_nll(logits_c, targets_c)
        cond_ok = bool(torch.isfinite(ce_cond))

        # token-entropy (optional diagnostic)
        if cond_ok:
            p_resp = torch.softmax(logits_c, dim=-1)
            ent_tok = -(p_resp * (p_resp.clamp_min(EPS)).log()).sum(-1)
            logit_entropy_mean = float(ent_tok.mean())
        else:
            logit_entropy_mean = float("nan")

        # ---------- UNCONDITIONAL ----------
        # For both LETTER and CONTENT we now use **no-options** unconditional scaffold (symmetric to question-only).
        if self.target == "letter":
            if option_label is None:
                m = re.match(r"\s*([A-Za-z])\s*\)", pack.response or "")
                option_label = (m.group(1).upper() if m else (ex.gold_letter or "A"))
            prompt_u, full_u, _ = prompts.uncond_letter(tok, option_label)
        else:
            prompt_u, full_u, _ = prompts.uncond_assistant_text(tok, pack.response)

        enc_u = tok(full_u, return_tensors="pt", add_special_tokens=False).to(model.device)
        enc_u_prompt = tok(prompt_u, return_tensors="pt", add_special_tokens=False)
        pref_len_u = int(enc_u_prompt.input_ids.shape[1])

        # map the same t0/span_len
        max_avail = enc_u.input_ids.shape[1] - (pref_len_u + t0)
        if max_avail <= 0:
            ce_uncond = torch.tensor(float('nan'))
            uncond_ok = False
        else:
            span_u = max(1, min(span_len, max_avail))
            sl_u = slice(pref_len_u - 1 + t0, pref_len_u - 1 + t0 + span_u)
            tl_u = slice(pref_len_u + t0,     pref_len_u + t0 + span_u)
            out_u = model(**enc_u, use_cache=False, return_dict=True)
            logits_u  = out_u.logits[0, sl_u, :]
            targets_u = enc_u.input_ids[0, tl_u]
            if logits_u.shape[0] != targets_u.shape[0]:
                L = min(logits_u.shape[0], targets_u.shape[0])
                logits_u  = logits_u[:L]
                targets_u = targets_u[:L]
            ce_uncond = utils._mean_nll(logits_u, targets_u)
            uncond_ok = bool(torch.isfinite(ce_uncond))

        # Use CE and PMI (no raw PPLs — they explode numerically)
        pmi = float((ce_uncond - ce_cond).item()) if (uncond_ok and cond_ok) else float("nan")

        out = {
            f"{self.prefix}ce_cond": float(ce_cond.item()) if cond_ok else float("nan"),
            f"{self.prefix}ce_uncond": float(ce_uncond.item()) if uncond_ok else float("nan"),
            f"{self.prefix}ce_logpmi": pmi,
            f"{self.prefix}logit_entropy_mean": logit_entropy_mean,
            f"{self.prefix}span_len": int(span_len),
            "llmc_prompt_len": int(prompt_len),
            "llmc_total_len": int(pack.total_len),
            "llmc_resp_len": int(resp_T),
        }
        # Backward-compat aliases expected by tables/aggregation (if any)
        if self.target == "content":
            out.update({
                "e_content_ce_logpmi": out[f"{self.prefix}ce_logpmi"],
                "e_content_ce_cond": out[f"{self.prefix}ce_cond"],
                "e_content_ce_uncond": out[f"{self.prefix}ce_uncond"],
            })
        else:
            out.update({
                "e_letter_ce_logpmi": out[f"{self.prefix}ce_logpmi"],
                "e_letter_ce_cond": out[f"{self.prefix}ce_cond"],
                "e_letter_ce_uncond": out[f"{self.prefix}ce_uncond"],
            })
        return out

        '''
                return {
                    f"{self.prefix}ce_cond": float(ce_cond.item()) if cond_ok else float("nan"),
                    f"{self.prefix}ce_uncond": float(ce_uncond.item()) if uncond_ok else float("nan"),
                    f"{self.prefix}ce_logpmi": pmi,  # PMI in nats (log-perplexity ratio)
                    f"{self.prefix}logit_entropy_mean": logit_entropy_mean,
                    f"{self.prefix}span_len": int(span_len),
                    "llmc_prompt_len": int(prompt_len),
                    "llmc_total_len": int(pack.total_len),
                    "llmc_resp_len": int(resp_T),
                }
        '''
