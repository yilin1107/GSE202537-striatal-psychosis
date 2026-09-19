"""A4: diagnosis-specific estimates (SCZ, BD) with 95% CI, SCZ-BD contrasts, genome-wide SCZ/BD concordance,
and BD power/minimum-detectable-effect analysis.

Outputs (out/):
  revision_A4_diagnosis_specific_scores.tsv      signature / marker scores: pooled, SCZ, BD, SCZ-BD (with and without run term)
  revision_A4_scz_bd_genomewide_concordance.tsv  per-region correlation and sign concordance of SCZ vs BD gene effects
  revision_A4_bd_power.tsv                       BD-arm power for SCZ-sized effects, minimum detectable logFC
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import revision_lib as rl


def ols_terms(y: np.ndarray, X: pd.DataFrame, terms: list[str], contrast: tuple[str, str] | None = None):
    Xm = X.to_numpy(dtype=float)
    n, p = Xm.shape
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    resid = y - Xm @ beta
    df = n - p
    s2 = float(resid @ resid) / df
    cov = np.linalg.pinv(Xm.T @ Xm) * s2
    cols = list(X.columns)
    rows = []
    tcrit = stats.t.ppf(0.975, df)
    for term in terms:
        j = cols.index(term)
        b, se = beta[j], np.sqrt(cov[j, j])
        t = b / se
        rows.append(dict(term=term, beta=b, se=se, ci_low=b - tcrit * se, ci_high=b + tcrit * se, t=t,
                         p=2 * stats.t.sf(abs(t), df), df=df))
    if contrast is not None:
        a, b_ = cols.index(contrast[0]), cols.index(contrast[1])
        c = np.zeros(p); c[a] = 1; c[b_] = -1
        est = float(c @ beta); se = float(np.sqrt(c @ cov @ c)); t = est / se
        rows.append(dict(term=f"{contrast[0]}_minus_{contrast[1]}", beta=est, se=se, ci_low=est - tcrit * se,
                         ci_high=est + tcrit * se, t=t, p=2 * stats.t.sf(abs(t), df), df=df))
    return rows


def score_models(scores: pd.DataFrame, meta: pd.DataFrame, score_type: str) -> pd.DataFrame:
    rows = []
    for region in rl.REGIONS:
        m = meta[meta.brain_region == region].reset_index(drop=True)
        designs = {
            "pooled": rl.region_design(m, diagnosis="pooled")[0],
            "separate": rl.region_design(m, diagnosis="separate")[0],
            "separate_run_fixed": rl.region_design(m, diagnosis="separate", run_fixed=True)[0],
            "pooled_run_fixed": rl.region_design(m, diagnosis="pooled", run_fixed=True)[0],
        }
        for score in scores.index:
            y_all = scores.loc[score]
            for model, X in designs.items():
                y = y_all.loc[X.index].to_numpy(dtype=float)
                if model.startswith("pooled"):
                    res = ols_terms(y, X, ["is_psychosis"])
                else:
                    res = ols_terms(y, X, ["is_scz", "is_bd_psychosis"], contrast=("is_scz", "is_bd_psychosis"))
                for r in res:
                    rows.append(dict(score_type=score_type, score=score, brain_region=region, model=model,
                                     n_scz=int(m.is_scz.sum()), n_bd=int(m.is_bd_psychosis.sum()),
                                     n_control=int((m.group == "Control").sum()), **r))
    df = pd.DataFrame(rows)
    df["fdr_within_region_term"] = df.groupby(["score_type", "brain_region", "model", "term"])["p"].transform(lambda x: rl.bh(x.values))
    return df


def genomewide_concordance(voom: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in rl.REGIONS:
        s = voom[(voom.brain_region == region) & (voom.contrast == "SCZ_vs_Control")].set_index("gene")
        b = voom[(voom.brain_region == region) & (voom.contrast == "BD_psychosis_vs_Control")].set_index("gene").loc[s.index]
        sig = s.index[s["adj.P.Val"] < 0.10]
        top500 = s["P.Value"].nsmallest(500).index
        def conc(idx):
            if len(idx) == 0:
                return np.nan, np.nan, np.nan
            agree = np.sign(s.loc[idx].logFC) == np.sign(b.loc[idx].logFC)
            k = int(agree.sum()); n = len(idx)
            return k / n, stats.binomtest(k, n, 0.5).pvalue, stats.spearmanr(s.loc[idx].logFC, b.loc[idx].logFC).statistic
        c_sig = conc(sig); c_top = conc(top500)
        rows.append(dict(brain_region=region, n_genes=len(s), n_scz_fdr10=len(sig),
                         spearman_t_all=stats.spearmanr(s.t, b.t).statistic,
                         spearman_logfc_all=stats.spearmanr(s.logFC, b.logFC).statistic,
                         sign_concordance_all=float(np.mean(np.sign(s.logFC) == np.sign(b.logFC))),
                         sign_concordance_scz_fdr10=c_sig[0], binom_p_scz_fdr10=c_sig[1], spearman_logfc_scz_fdr10=c_sig[2],
                         sign_concordance_top500=c_top[0], binom_p_top500=c_top[1], spearman_logfc_top500=c_top[2],
                         median_abs_logfc_scz_fdr10_in_scz=float(s.loc[sig].logFC.abs().median()) if len(sig) else np.nan,
                         median_abs_logfc_scz_fdr10_in_bd=float(b.loc[sig].logFC.abs().median()) if len(sig) else np.nan,
                         slope_bd_on_scz_fdr10=float(np.polyfit(s.loc[sig].logFC, b.loc[sig].logFC, 1)[0]) if len(sig) > 2 else np.nan))
    return pd.DataFrame(rows)


def bd_power(voom: pd.DataFrame) -> pd.DataFrame:
    """Power of the BD arm to detect SCZ-sized effects, using the fitted voom SEs."""
    rows = []
    for region in rl.REGIONS:
        E = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_E.npy"); W = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_W.npy")
        X = pd.read_csv(rl.INTERMEDIATE / f"A1_fit_{region}_design.tsv", sep="\t", index_col=0)
        coef, su, sigma, df_res, _ = rl.lm_fit(E, X.to_numpy(), weights=W)
        cols = list(X.columns)
        eb, var_post, var_prior, df_prior = rl.ebayes(coef, su, sigma, df_res, [cols.index("is_scz"), cols.index("is_bd_psychosis")])
        se_scz = su[:, cols.index("is_scz")] * np.sqrt(var_post)
        se_bd = su[:, cols.index("is_bd_psychosis")] * np.sqrt(var_post)
        df_total = eb[cols.index("is_scz")]["df_total"]
        genes = pd.Index(pd.read_csv(rl.PROCESSED / "gse202537_symbol_counts_filtered.tsv", sep="\t", index_col=0).index)
        s = voom[(voom.brain_region == region) & (voom.contrast == "SCZ_vs_Control")].set_index("gene").loc[genes]
        sig = np.where(s["adj.P.Val"].values < 0.10)[0]
        alpha = 0.05
        tcrit = stats.t.ppf(1 - alpha / 2, df_total)
        # FDR-equivalent nominal threshold: largest P among FDR<0.10 genes in the SCZ analysis (if any)
        p_thr = float(s["P.Value"].values[sig].max()) if len(sig) else np.nan
        tcrit_fdr = stats.t.ppf(1 - p_thr / 2, df_total) if len(sig) else np.nan
        def power(delta, se, tc):
            ncp = delta / se
            return stats.nct.sf(tc, df_total, ncp) + stats.nct.cdf(-tc, df_total, ncp)
        mde_bd = (tcrit + stats.t.ppf(0.8, df_total)) * se_bd   # 80% power at alpha 0.05
        mde_scz = (tcrit + stats.t.ppf(0.8, df_total)) * se_scz
        row = dict(brain_region=region, n_bd=int(X["is_bd_psychosis"].sum()), n_scz=int(X["is_scz"].sum()),
                   n_control=int(((X["is_scz"] == 0) & (X["is_bd_psychosis"] == 0)).sum()), df_total=float(df_total),
                   median_se_scz=float(np.median(se_scz)), median_se_bd=float(np.median(se_bd)),
                   se_ratio_bd_over_scz=float(np.median(se_bd / se_scz)),
                   mde_logfc_80pct_alpha05_scz_median=float(np.median(mde_scz)),
                   mde_logfc_80pct_alpha05_bd_median=float(np.median(mde_bd)),
                   n_scz_fdr10=len(sig), scz_fdr10_p_threshold=p_thr)
        if len(sig):
            delta = np.abs(s.logFC.values[sig])
            pw05 = power(delta, se_bd[sig], tcrit)
            pwfdr = power(delta, se_bd[sig], tcrit_fdr)
            row.update(median_abs_logfc_scz_fdr10=float(np.median(delta)),
                       bd_power_alpha05_median=float(np.median(pw05)), bd_power_alpha05_frac_ge80=float(np.mean(pw05 >= 0.8)),
                       bd_power_fdr_equiv_median=float(np.median(pwfdr)), bd_power_fdr_equiv_frac_ge80=float(np.mean(pwfdr >= 0.8)),
                       frac_scz_fdr10_effects_below_bd_mde=float(np.mean(delta < mde_bd[sig])))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    meta = rl.load_metadata()
    sig_scores = pd.read_csv(rl.PROCESSED / "gse202537_molecular_signature_scores.csv", index_col=0)
    marker_scores = pd.read_csv(rl.PROCESSED / "gse202537_cell_type_marker_scores.csv", index_col=0)
    ext_scores = pd.read_csv(rl.PROCESSED / "gse202537_hpa_snbrain_cell_type_scores.csv", index_col=0)
    parts = [score_models(sig_scores, meta, "directional_signature"),
             score_models(marker_scores, meta, "internal_marker_score"),
             score_models(ext_scores, meta, "external_hpa_marker_score")]
    scores = pd.concat(parts, ignore_index=True)
    # validation against the original pooled results
    orig = pd.read_csv(rl.TABLES / "figure4_molecular_signature_results.tsv", sep="\t")
    chk = scores[(scores.score_type == "directional_signature") & (scores.model == "pooled")].merge(
        orig, left_on=["score", "brain_region"], right_on=["signature", "brain_region"])
    print("max |beta diff| vs original pooled signature table:", float((chk.beta - chk.beta_psychosis).abs().max()))
    scores.to_csv(rl.OUT / "revision_A4_diagnosis_specific_scores.tsv", sep="\t", index=False)
    voom = pd.read_csv(rl.OUT / "revision_A1_voom_gene_results.tsv", sep="\t")
    conc = genomewide_concordance(voom)
    conc.to_csv(rl.OUT / "revision_A4_scz_bd_genomewide_concordance.tsv", sep="\t", index=False)
    pw = bd_power(voom)
    pw.to_csv(rl.OUT / "revision_A4_bd_power.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(conc.round(3).to_string()); print(pw.round(3).to_string())
    sep = scores[(scores.score_type == "directional_signature") & (scores.model == "separate")]
    print(sep.pivot_table(index=["score", "brain_region"], columns="term", values=["beta", "p"]).round(3).to_string())


if __name__ == "__main__":
    main()
