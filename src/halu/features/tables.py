# Halu/features/tables.py
from __future__ import annotations
from typing import List
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from halu.data.base import LETTERS
from halu.utils.math import top2_gap, entropy_from_prob

def _letters_in_rec(rec: dict) -> List[str]:
    """
    Infer which option letters are present in this record.
    Priority:
      1) Any letter-prefixed metrics (A_*, B_*, ...)
      2) Keys of fcm_letter_prob_map
      3) Explicit fcm_prob_letters list
    Fall back to A..D if nothing found.
    """
    found: List[str] = []
    # 1) From letter-prefixed fields
    pref = [L for L in LETTERS if any(k.startswith(f"{L}_") for k in rec.keys())]
    found.extend(pref)
    # 2) From map
    fmap = rec.get("fcm_letter_prob_map")
    if isinstance(fmap, dict):
        found.extend([str(k).upper() for k in fmap.keys() if isinstance(k, str)])
    # 3) From ordered list (paired with fcm_letter_probs)
    flist = rec.get("fcm_prob_letters")
    if isinstance(flist, (list, tuple)):
        found.extend([str(k).upper() for k in flist])
    # Dedup in order & keep only known letters
    seen = set()
    out: List[str] = []
    for L in found:
        if L in LETTERS and L not in seen:
            seen.add(L); out.append(L)
    return out or ["A","B","C","D"]

def _safe_float(x, default=np.nan):
    try:
        v = float(x);  return v if np.isfinite(v) else default
    except Exception:
        return default

def build_option_table(recs: List[dict]) -> pd.DataFrame:
    rows = []
    for r in tqdm(recs, desc="build option rows"):
        qid = r["qid"]; dataset = r.get("dataset","")
        vanilla = str(r.get("model_pred") or r.get("pred_letter") or "").strip().upper()
        gold = str(r.get("gold_letter","")).strip().upper()
        letters = _letters_in_rec(r)
        fcm_probs = r.get("fcm_letter_probs", None)

        for i, L in enumerate(letters):
            def g_alias(name: str, default=np.nan):
                alias_map = {
                    "pmil": ["llmc_letter_ce_logpmi", "pmil"],
                    "pmic": ["llmc_content_ce_logpmi", "pmic"],
                    "e_letter": ["llmc_letter_logit_entropy_mean", "e_letter"],
                    "e_content": ["llmc_content_logit_entropy_mean", "e_content"],
                    "icrp_reliable": ["icrp_reliable"],
                    "icrp_prompt_mass_mean": ["icrp_prompt_mass_mean", "icrp_pmass"],
                    "icrp_pool_attn_cons": ["icrp_pool_attn_cons", "icrp_attn"],
                    "icrp_time_delta_mean": ["icrp_time_delta_mean"],
                    "icrp_layer_delta_mean": ["icrp_layer_delta_mean"],
                    "icrp_interlayer_cos_mean": ["icrp_interlayer_cos_mean"],
                    "icrp_temporal_cos_mean": ["icrp_temporal_cos_mean"],
                    "icrp_resp_len": ["icrp_resp_len"],
                }
                for suf in alias_map.get(name, [name]):
                    key = f"{L}_{suf}"
                    if key in r and r[key] is not None:
                        return _safe_float(r[key], default)
                return default

            # FCM sources (all optional):
            fcm_map = r.get("fcm_letter_prob_map")  # {'A':0.1, 'B':...}
            fcm_letters = r.get("fcm_prob_letters")  # ['B','C','A',...], aligned with fcm_probs

            def _fcm_for(L, i):
                if isinstance(fcm_map, dict) and L in fcm_map:
                    return _safe_float(fcm_map[L])
                if isinstance(fcm_probs, (list, tuple)):
                    if isinstance(fcm_letters, (list, tuple)) and len(fcm_letters) == len(fcm_probs):
                        try:
                            j = [s.upper() for s in fcm_letters].index(L)
                            return _safe_float(fcm_probs[j])
                        except ValueError:
                            return np.nan
                    if i < len(fcm_probs):
                        return _safe_float(fcm_probs[i])
                return np.nan

            fcm_prob = _fcm_for(L, i)

            rows.append(dict(
                qid=qid, dataset=dataset, letter=L, gold=gold, vanilla=vanilla,
                pmil=g_alias("pmil"),
                pmic = g_alias("pmic"),
                e_letter = g_alias("e_letter"),
                e_content = g_alias("e_content"),
                icrp_reliable = g_alias("icrp_reliable"),
                icrp_rel = g_alias("icrp_reliable"),
                icrp_prompt_mass_mean = g_alias("icrp_prompt_mass_mean"),
                icrp_pool_attn_cons = g_alias("icrp_pool_attn_cons"),
                icrp_time_delta_mean = g_alias("icrp_time_delta_mean"),
                icrp_layer_delta_mean = g_alias("icrp_layer_delta_mean"),
                icrp_interlayer_cos_mean = g_alias("icrp_interlayer_cos_mean"),
                icrp_temporal_cos_mean = g_alias("icrp_temporal_cos_mean"),
                icrp_resp_len = g_alias("icrp_resp_len"),
                icrp_pmass = g_alias("icrp_prompt_mass_mean"),
                icrp_attn = g_alias("icrp_pool_attn_cons"),
                fcm_prob=fcm_prob,
            ))
    OPT = pd.DataFrame(rows)
    if OPT.empty:
        raise ValueError("No per-option rows built; check input.")

    for _col in ["pmil","pmic","e_letter","e_content","icrp_reliable","fcm_prob"]:
        if _col in OPT and pd.api.types.is_numeric_dtype(OPT[_col]):
            asc = (_col in ["e_letter","e_content"])
            OPT[_col + "_rank"] = OPT.groupby("qid")[_col].transform(lambda s: s.rank(ascending=asc, method="min"))
    OPT["y_opt"] = (OPT["letter"].str.upper() == OPT["gold"].str.upper()).astype(int)
    return OPT

def _gini(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = np.sort(p / s)
    n = len(p)
    return float((2.0 * (np.arange(1, n + 1) * p).sum() - (n + 1)) / n)

def _mk_popt_for_group(g: pd.DataFrame) -> pd.Series:
    f = pd.to_numeric(g.get("fcm_prob"), errors="coerce")
    if f.notna().any():
        f = f.fillna(0.0); s = f.sum()
        if s > 0: return f / s
    pmic = pd.to_numeric(g.get("pmic"), errors="coerce"); pmic = pmic.fillna(pmic.median())
    pmil = pd.to_numeric(g.get("pmil"), errors="coerce"); pmil = pmil.fillna(pmil.median())
    sc = 0.6 * pmic + 0.4 * pmil
    if sc.notna().any():
        z = sc - sc.max(); p = np.exp(z); s = float(p.sum())
        if s > 0: return pd.Series(p / s, index=g.index)
    n = len(g); return pd.Series(np.ones(n, dtype=float) / max(n, 1), index=g.index)

def collapse_option_side_features(OPT_with_p: pd.DataFrame) -> pd.DataFrame:
    df = OPT_with_p.copy()
    if "p_opt" not in df.columns or df["p_opt"].isna().all():
        df["p_opt"] = np.nan
        for _, g in df.groupby("qid"):
            df.loc[g.index, "p_opt"] = _mk_popt_for_group(g)

    def _renorm_group(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").astype(float)
        s = s.where(np.isfinite(s), 0.0)
        tot = float(s.sum())
        if tot > 0:
            return s / tot
        # all zero/NaN → flag unusable distribution
        return pd.Series(np.nan, index=s.index, dtype=float)

    df["p_opt"] = df.groupby("qid", group_keys=False)["p_opt"].apply(_renorm_group)

    outs = []
    for qid, grp0 in tqdm(df.groupby("qid"), desc="collapse per-question"):
        grp = grp0.copy()
        vanilla = str(grp["vanilla"].iloc[0]).strip().upper()
        gold    = str(grp["gold"].iloc[0]).strip().upper()
        grp["letter"] = grp["letter"].astype(str).str.upper()
        grp = grp.sort_values("p_opt", ascending=False)
        chose = grp[grp["letter"] == vanilla]
        p_vec = pd.to_numeric(grp["p_opt"], errors="coerce").astype(float).values
        p_top = float(p_vec[0]) if len(p_vec) else np.nan
        p_second = float(p_vec[1]) if len(p_vec) > 1 else np.nan
        p_gap = p_top - p_second if np.isfinite(p_top) and np.isfinite(p_second) else np.nan
        p_entropy = entropy_from_prob(p_vec); p_gini = _gini(p_vec)
        p_chosen = float(chose["p_opt"].iloc[0]) if not chose.empty else np.nan
        ranks = grp["p_opt"].rank(ascending=False, method="min")
        chosen_rank = float(ranks.loc[chose.index].iloc[0]) if not chose.empty else np.nan
        L_top = str(grp["letter"].iloc[0]) if len(grp) else ""

        def _col(name: str) -> pd.Series:
            if name in grp: return pd.to_numeric(grp[name], errors="coerce").astype(float)
            aliases = {"icrp_prompt_mass_mean":["icrp_pmass"], "icrp_pool_attn_cons":["icrp_attn"], "icrp_rel":["icrp_reliable"]}
            for alt in aliases.get(name, []):
                if alt in grp: return pd.to_numeric(grp[alt], errors="coerce").astype(float)
            return pd.Series(np.nan, index=grp.index, dtype=float)

        def chosen_val(series: pd.Series):
            if chose.empty: return np.nan
            s = pd.to_numeric(series.loc[chose.index], errors="coerce"); s = s[np.isfinite(s)]
            return float(s.iloc[0]) if len(s) else np.nan

        def max_other(series: pd.Series):
            if chose.empty: return np.nan
            s = pd.to_numeric(series.drop(index=chose.index), errors="coerce"); s = s[np.isfinite(s)]
            return float(s.max()) if len(s) else np.nan

        pmic = _col("pmic"); pmil = _col("pmil")
        pmic_gap = top2_gap(pmic.values); pmil_gap = top2_gap(pmil.values)

        fcm = _col("fcm_prob").values
        fcm_entropy = entropy_from_prob(fcm) if np.isfinite(fcm).any() else np.nan
        fcm_gini    = _gini(fcm) if np.isfinite(fcm).any() else np.nan
        letters_cov = float((fcm > 0.05).sum()) if np.isfinite(fcm).any() else np.nan
        if np.isfinite(fcm).any():
            fcm_top_idx = int(np.nanargmax(fcm))
            fcm_top_letter = str(grp["letter"].iloc[fcm_top_idx]) if len(grp) else ""
            mass_on_letters = float(np.nansum(fcm))
            mass_gap = (mass_on_letters - float(np.nanmax(fcm)))
        else:
            fcm_top_letter, mass_gap = "", np.nan

        outs.append(dict(
            qid=qid, gold=gold, chosen=vanilla,
            p_opt_chosen=p_chosen, p_opt_top=p_top, p_opt_second=p_second, p_opt_gap=p_gap,
            p_opt_rank_chosen=chosen_rank, opt_top_letter=L_top,
            delta_pmil_chosen_vs_othermax=(chosen_val(pmil) - max_other(pmil)) if np.isfinite(chosen_val(pmil)) else np.nan,
            delta_pmic_chosen_vs_othermax=(chosen_val(pmic) - max_other(pmic)) if np.isfinite(chosen_val(pmic)) else np.nan,
            delta_icrp_rel_chosen_vs_othermax=(chosen_val(_col("icrp_rel")) - max_other(_col("icrp_rel"))),
            delta_e_letter_chosen_vs_othermax=-(chosen_val(_col("e_letter")) - max_other(_col("e_letter"))),
            delta_e_content_chosen_vs_othermax=-(chosen_val(_col("e_content")) - max_other(_col("e_content"))),
            delta_fcm_prob_chosen_vs_othermax=(chosen_val(_col("fcm_prob")) - max_other(_col("fcm_prob"))),
            p_opt_entropy=p_entropy, p_opt_gini=p_gini,
            pmic_gap_top1_top2=pmic_gap, pmil_gap_top1_top2=pmil_gap,
            pmic_rank_chosen=float(pmic.rank(ascending=False, method="min").loc[chose.index].iloc[0]) if not chose.empty else np.nan,
            pmil_rank_chosen=float(pmil.rank(ascending=False, method="min").loc[chose.index].iloc[0]) if not chose.empty else np.nan,
            fcm_entropy=fcm_entropy, fcm_gini=fcm_gini, fcm_letters_coverage=letters_cov, fcm_mass_gap=mass_gap,
            fcm_top_letter=fcm_top_letter,
        ))
    OUT = pd.DataFrame(outs)
    return OUT