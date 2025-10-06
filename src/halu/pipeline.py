# halu/pipeline.py (only the modified parts)
from __future__ import annotations
from typing import List, Dict, Set
import numpy as np
import torch
from .core.types import MCQExample, ForwardPack
from .core.runner import HFRunner
from .core import utils
from features.metrics.icr_probe import ICRProbeMetric
from features.metrics.llmcheck import LLMCheckMetric
from features.metrics.fcm import FCMMetric


def _question_only_scaffold(tok, question: str) -> tuple[str, int]:
    sys = "You are a helpful, factual assistant."
    usr = f"Question: {question}\nAnswer:"
    if hasattr(tok, "apply_chat_template"):
        msgs = [{"role":"system","content":sys},{"role":"user","content":usr}]
        q_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        q_prompt = sys + "\n\n" + usr
    q_len = int(tok(q_prompt, return_tensors="pt").input_ids.shape[1])
    return q_prompt, q_len

class MetricsPipeline:
    def __init__(self, runner: HFRunner, *, use_icr: bool = True, use_llmcheck: bool = True, use_fcm: bool = True):
        self.runner = runner
        self.icr = ICRProbeMetric() if use_icr else None

        if use_llmcheck:
            self.llmc_letter  = LLMCheckMetric(runner=runner, target="letter",  span_mode="first")
            self.llmc_content = LLMCheckMetric(runner=runner, target="content", span_mode="all")
        else:
            self.llmc_letter = self.llmc_content = None

        self.fcm = FCMMetric(runner) if use_fcm else None

    def _needs_union(self) -> Set[str]:
        needs: Set[str] = set()
        for m in (self.icr, self.llmc_letter, self.llmc_content):
            if m is not None:
                needs.update(m.needs)
        return needs

    # inside class MetricsPipeline
    def _batch_forward_options(self, prompt: str, responses: list[str], need: Set[str]) -> list[ForwardPack]:
        tok, model = self.runner.tok, self.runner.model

        # Normalize "needs"
        need = {("hidden_layers" if n in {"hidden", "hidden_layers"} else
                "attn_layers"  if n in {"attn", "attn_layers"} else
                n) for n in (need or set())}

        want_attn   = ("attn_layers" in need)
        want_hidden = ("hidden_layers" in need)
        want_logits = ("logits" in need) or ("logits_full" in need)

        texts = [prompt + r for r in responses]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)

        # Shared prompt length (no padding needed)
        prompt_len = int(tok(prompt, return_tensors="pt").input_ids.shape[1])

        with torch.inference_mode():
            out = model(**enc,
                        output_attentions=want_attn,
                        output_hidden_states=want_hidden,
                        use_cache=False,
                        return_dict=True)

        # Per-sample true lengths (exclude right padding)
        total_lens = enc.attention_mask.sum(dim=1).tolist()  # List[int]
        B = len(responses)

        packs: list[ForwardPack] = []
        for i in range(B):
            total_len = int(total_lens[i])
            resp_len  = int(total_len - prompt_len)
            assert resp_len > 0, "Empty response region in batch forward."

            # ---- logits (slice once per sample) ----
            logits_full = None
            logits_resp = None
            if want_logits:
                lf = out.logits[i]  # [T, V]
                if self.runner.keep_logits_full or ("logits_full" in need):
                    logits_full = self.runner._place(lf, dtype=self.runner.store_dtype_logits)
                # align (see comment #18)
                if prompt_len == 0:
                    logits_resp = lf[: total_len - 1][-resp_len:]
                else:
                    logits_resp = lf[prompt_len - 1 : total_len - 1]
                logits_resp = self.runner._place(logits_resp, dtype=self.runner.store_dtype_logits)

            # ---- attention: only the layer we intend to use (item #6) ----
            attn_layers = None
            if want_attn and out.attentions is not None:
                last = out.attentions[-1][i]  # [H,T,T]
                attn_layers = [ self.runner._place(last, dtype=self.runner.store_dtype_states) ]

            # ---- hidden states (all layers; or you can subset here) ----
            hidden_layers = None
            hidden_embed  = None
            if want_hidden and out.hidden_states is not None:
                hs = out.hidden_states
                # embeddings if runner.include_embeddings
                if self.runner.include_embeddings:
                    hidden_embed = self.runner._place(hs[0][i], dtype=self.runner.store_dtype_states)
                hidden_layers = [ self.runner._place(hs[j][i], dtype=self.runner.store_dtype_states)
                                  for j in range(1, len(hs)) ]

            packs.append(ForwardPack(
                prompt=prompt,
                response=responses[i],
                prompt_len=prompt_len,
                total_len=total_len,
                resp_len=resp_len,
                logits_resp=logits_resp,
                logits_full=logits_full,
                attn_layers=attn_layers,
                hidden_layers=hidden_layers,
                hidden_embed=hidden_embed,
            ))
        return packs

    @torch.no_grad()
    def example_to_row(self, ex: MCQExample) -> Dict:
        tok = self.runner.tok
        prompt = utils.build_prompt(ex, tokenizer=tok)
        fcm_feats = self.fcm.compute(ex) if self.fcm is not None else {}

        per_option_feats: List[Dict] = []
        option_labels = [o.label for o in ex.options]
        needs = self._needs_union()
        q_prompt, q_prompt_len = _question_only_scaffold(tok, ex.question)

        """for o in ex.options:
            response = f"{o.label}) {o.text}"
            pack = self.runner.prompt_response_forward(prompt, response, need=needs)
            feats = {}
            if self.llmc_letter  is not None: feats.update(self.llmc_letter.compute(pack, ex, option_label=o.label))
            if self.llmc_content is not None: feats.update(self.llmc_content.compute(pack, ex, option_label=o.label, cached_q_prompt=q_prompt, cached_q_prompt_len=q_prompt_len))
            if self.icr is not None:          feats.update(self.icr.compute(pack, ex))
            per_option_feats.append(feats)"""

        responses = [f"{o.label}) {o.text}" for o in ex.options]
        packs = self._batch_forward_options(prompt, responses, needs)

        for o, pack in zip(ex.options, packs):
            feats = {}
            if self.llmc_letter  is not None: feats.update(self.llmc_letter.compute(pack, ex, option_label=o.label))
            if self.llmc_content is not None: feats.update(self.llmc_content.compute(pack, ex, option_label=o.label, cached_q_prompt=q_prompt, cached_q_prompt_len=q_prompt_len))
            if self.icr is not None:          feats.update(self.icr.compute(pack, ex))
            per_option_feats.append(feats)

        # ---- Per-question centering for content PMI ----
        # collect in order of option_labels
        pmis = []
        for lab, feats in zip(option_labels, per_option_feats):
            pmis.append(feats.get("llmc_content_ce_logpmi", np.nan))
        pmis = np.array(pmis, dtype=np.float32)
        mu = np.nanmean(pmis) if np.isfinite(pmis).any() else 0.0
        centered = pmis - mu
        # (top1 - top2) gap
        # we want max minus second max (ignoring nans)
        finite = np.isfinite(centered)
        gap = float("nan")
        if finite.any():
            vals = np.sort(centered[finite])
            if vals.size >= 2:
                gap = float(vals[-1] - vals[-2])
            elif vals.size == 1:
                gap = float(vals[-1])

        # write centered features back into the row as <L>_llmc_content_ce_logpmi_ctr
        row = utils.get_single_row_features_from_option_metrics(per_option_feats, option_labels)
        for lab, val in zip(option_labels, centered):
            row[f"{lab}_llmc_content_ce_logpmi_ctr"] = float(val) if np.isfinite(val) else float("nan")
        row["llmc_content_ce_logpmi_gap_top1_top2"] = gap

        lltr = []
        for lab, feats in zip(option_labels, per_option_feats):
            lltr.append(feats.get("llmc_letter_ce_logpmi", np.nan))
        lltr = np.array(lltr, dtype=np.float32)
        mu_l = np.nanmean(lltr) if np.isfinite(lltr).any() else 0.0
        centered_l = lltr - mu_l

        finite_l = np.isfinite(centered_l)
        gap_l = float("nan")
        if finite_l.any():
            vals = np.sort(centered_l[finite_l])
            if vals.size >= 2:
                gap_l = float(vals[-1] - vals[-2])
            elif vals.size == 1:
                gap_l = float(vals[-1])

        for lab, val in zip(option_labels, centered_l):
            row[f"{lab}_llmc_letter_ce_logpmi_ctr"] = float(val) if np.isfinite(val) else float("nan")
        row["llmc_letter_ce_logpmi_gap_top1_top2"] = gap_l

        model_pred = fcm_feats.get("pred_letter") or ""
        return dict(
            qid=ex.qid, dataset=ex.dataset, question=ex.question,
            gold_letter=(ex.gold_letter or "").upper(),
            model_pred=model_pred,
            is_hallucination=int((model_pred or "").upper() != (ex.gold_letter or "").upper()) if ex.gold_letter else 0,
            **row, **fcm_feats,
        )
