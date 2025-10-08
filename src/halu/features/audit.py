def audit_features(df, strict=False):
    def nz(c): return float(df[c].notna().mean()) if c in df else 0.0
    rep = {"llmc_letter_cov": nz("e_letter_ce_logpmi"), "llmc_content_cov": nz("e_content_ce_logpmi"),
           "pmil_cov": nz("pmil_gap_top1_top2"), "pmic_cov": nz("pmic_gap_top1_top2"),
           "icrp_cov": nz("delta_icrp_rel_chosen_vs_othermax"), "fcm_cov": nz("fcm_entropy"),
           "popt_uniform_frac": float((df.get("p_opt_gap", 0).fillna(0.0) == 0.0).mean()), "warnings": []}
    if rep["llmc_letter_cov"]  < 0.9: rep["warnings"].append("LLMCheck(letter) low")
    if rep["llmc_content_cov"] < 0.5: rep["warnings"].append("LLMCheck(content) low")
    if rep["pmil_cov"]         < 0.9: rep["warnings"].append("PMI(letter) low")
    if rep["pmic_cov"]         < 0.5: rep["warnings"].append("PMI(content) low")
    if rep["icrp_cov"]         < 0.7: rep["warnings"].append("ICR probe low")
    if rep["fcm_cov"]          < 0.1: rep["warnings"].append("FCM coverage near zero")
    if rep["popt_uniform_frac"]> 0.5: rep["warnings"].append("Option probs almost uniform")
    if strict and rep["warnings"]:
        raise RuntimeError(f"Feature audit failed: {rep}")
    return rep