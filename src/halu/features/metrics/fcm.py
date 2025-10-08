# src/halu/features/metrics/fcm.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
from halu.core.types import MCQExample
from halu.core import utils
from halu.core import prompts

class FCMMetric:
    def __init__(self, runner, temperature: float = 1.0, n_votes: int = 16):
        self.runner = runner
        self.temperature = temperature
        self.n_votes = n_votes

    @torch.no_grad()
    def compute(self, ex: MCQExample) -> Dict[str, Any]:
        model = self.runner.model
        tokenizer = self.runner.tok
        model.eval()
        options_text = utils.options_text_from_ex(ex)
        letters = utils.labels_from_ex(ex)
        prompt, _ = prompts.q_with_options(tokenizer, ex.question, options_text)
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        logits = model(**enc, use_cache=False, return_dict=True).logits[0, -1, :]  # [V]

        p_letters, mass_on_letters, buckets = utils._letter_dist_fullsoftmax(logits, letters, tokenizer, self.temperature)
        gini = float(1.0 - float((p_letters * p_letters).sum()))
        letters_with_bucket = sum(1 for b in buckets if len(b) > 0)

        if p_letters.sum() <= 0:
            gen = model.generate(**enc, do_sample=False, temperature=0.0, max_new_tokens=1,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                 eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id)
            out_txt = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            first = (out_txt[:1].upper() if out_txt else "")
            p_letters = np.zeros(len(letters), dtype=np.float32)
            mass_on_letters = 0.0
            if first in letters:
                p_letters[letters.index(first)] = 1.0
                mass_on_letters = 1.0

        p = np.asarray(p_letters, dtype=np.float32)
        p_sorted = np.sort(p)[::-1]
        top1 = float(p_sorted[0]) if p.size >= 1 else 0.0
        top2 = float(p_sorted[1]) if p.size >= 2 else 0.0
        entropy = float(max(0.0, -(p * np.log(p + 1e-8)).sum()))
        rng = np.random.default_rng(utils.SEED)
        draws = rng.choice(len(p), size=self.n_votes, p=(p / p.sum()) if p.sum() > 0 else None)
        counts = np.bincount(draws, minlength=len(p)).astype(np.float32)
        vote_probs = counts / max(1, self.n_votes)
        vote_entropy = float(max(0.0, -(vote_probs * np.log(vote_probs + 1e-8)).sum()))
        top_share = float(vote_probs.max())

        pred_idx = int(p.argmax()) if p.sum() > 0 else 0
        pred_letter = letters[pred_idx]

        letter_max = np.array([float(logits[b].max().item()) if b else -1e9 for b in buckets], dtype=np.float32)
        letter_lse = np.array([float(torch.logsumexp(logits[b], dim=0).item()) if b else -1e9 for b in buckets], dtype=np.float32)
        mx_sorted = np.sort(letter_max)
        lse_sorted = np.sort(letter_lse)
        mx1, mx2 = mx_sorted[-1], mx_sorted[-2]
        ls1, ls2 = lse_sorted[-1], lse_sorted[-2]

        out = {
            "pred_letter": pred_letter.upper(),

            # New rich diagnostics
            "fcm_mass_on_letters": float(mass_on_letters),
            "fcm_letter_probs": p.tolist(),
            "fcm_letter_p_top1": top1,
            "fcm_letter_p_top2": top2,
            "fcm_letter_margin": float(top1 - top2),
            "fcm_letter_entropy": entropy,
            "fcm_vote_top_share": top_share,
            "fcm_vote_entropy": vote_entropy,
            "fcm_letter_logit_top1": float(mx1),
            "fcm_letter_logit_top2": float(mx2),
            "fcm_letter_logit_margin": float(mx1 - mx2),
            "fcm_letter_lse_top1": float(ls1),
            "fcm_letter_lse_top2": float(ls2),
            "fcm_letter_lse_margin": float(ls1 - ls2),

            # ---- Legacy aliases expected by your tables ----
            "fcm_top_letter": pred_letter.upper(),
            "fcm_mass_gap": float(top1 - top2),
            "fcm_letters_coverage": float(letters_with_bucket),  # was 4.0/5.0 in your samples
            "fcm_entropy": float(entropy),  # entropy over p_letters
            "fcm_gini": float(gini),
        }
        return out
'''        out = {
            "pred_letter": pred_letter.upper(),
            "fcm_mass_on_letters": float(mass_on_letters),
            "fcm_letter_probs": p.tolist(),
            "fcm_letter_p_top1": top1,
            "fcm_letter_p_top2": top2,
            "fcm_letter_margin": float(top1 - top2),
            "fcm_letter_entropy": entropy,
            "fcm_vote_top_share": top_share,
            "fcm_vote_entropy": vote_entropy,
            "fcm_letter_logit_top1": float(mx1),
            "fcm_letter_logit_top2": float(mx2),
            "fcm_letter_logit_margin": float(mx1 - mx2),
            "fcm_letter_lse_top1": float(ls1),
            "fcm_letter_lse_top2": float(ls2),
            "fcm_letter_lse_margin": float(ls1 - ls2),
        }
        return out'''
