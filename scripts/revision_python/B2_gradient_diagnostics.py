"""B2: why does the schizophrenia gene-level burden rise from NAc to putamen? Diagnostics.

Outputs (out/):
  revision_B2_region_gradient_diagnostics.tsv   per region: n, pi0 (Storey), P-value histogram bins, residual variance, RIN/depth,
                                                 voom & OLS FDR<0.10 counts, cross-region t correlations
  revision_B2_cross_region_concordance.tsv      concordance of putamen SCZ FDR genes in caudate / NAc (and vice versa)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import revision_lib as rl


def storey_pi0(p: np.ndarray, lambdas=np.arange(0.05, 0.96, 0.05)):
    p = np.asarray(p)
    pi0s = np.array([np.mean(p > l) / (1 - l) for l in lambdas])
    # Storey & Tibshirani smoother: cubic spline through pi0(lambda), evaluated at max lambda
    try:
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(lambdas, pi0s, k=3, s=len(lambdas) * 0.02)
        pi0 = float(min(1.0, spl(lambdas[-1])))
    except Exception:
        pi0 = float(min(1.0, pi0s[-1]))
    return pi0, float(min(1.0, pi0s[np.argmin(np.abs(lambdas - 0.5))]))


def main():
    voom = pd.read_csv(rl.OUT / "revision_A1_voom_gene_results.tsv", sep="\t")
    meta = rl.load_metadata()
    rows = []
    tstats = {}
    for region in rl.REGIONS:
        s = voom[(voom.brain_region == region) & (voom.contrast == "SCZ_vs_Control")].set_index("gene")
        b = voom[(voom.brain_region == region) & (voom.contrast == "BD_psychosis_vs_Control")].set_index("gene")
        tstats[region] = s.t
        pi0_smooth, pi0_half = storey_pi0(s["P.Value"].values)
        pi0_ols_smooth, _ = storey_pi0(s["ols_p"].values)
        hist, _ = np.histogram(s["P.Value"].values, bins=np.linspace(0, 1, 11))
        E = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_E.npy"); W = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_W.npy")
        X = pd.read_csv(rl.INTERMEDIATE / f"A1_fit_{region}_design.tsv", sep="\t", index_col=0)
        coef, su, sigma, df_res, resid = rl.lm_fit(E, X.to_numpy(), weights=W)
        m = meta.set_index("sample_key").loc[X.index]
        rows.append(dict(
            brain_region=region, n_samples=len(X), n_scz=int(X.is_scz.sum()), n_bd=int(X.is_bd_psychosis.sum()),
            n_control=int(((X.is_scz == 0) & (X.is_bd_psychosis == 0)).sum()),
            voom_scz_fdr10=int((s["adj.P.Val"] < 0.10).sum()), ols_scz_fdr10=int((s.ols_fdr < 0.10).sum()),
            voom_scz_nominal_p05=int((s["P.Value"] < 0.05).sum()), expected_p05_under_null=int(0.05 * len(s)),
            pi0_storey_smoother_voom=pi0_smooth, pi0_lambda05_voom=pi0_half, pi0_storey_smoother_ols=pi0_ols_smooth,
            est_n_nonnull_voom=int(round((1 - pi0_smooth) * len(s))),
            p_hist_bin1_frac=hist[0] / len(s), p_hist_bin10_frac=hist[-1] / len(s),
            median_residual_sd_voom=float(np.median(sigma)), median_abs_t_scz=float(np.median(np.abs(s.t))),
            mean_abs_logfc_scz=float(np.mean(np.abs(s.logFC))),
            median_rin=float(m.rin.median()), median_pmi=float(m.pmi.median()), median_depth_million=float(m.count_depth_million.median()),
            median_ph=float(m.ph.median()), n_runs=int(m.sequence_id.nunique()),
            scz_control_rin_p=stats.mannwhitneyu(m.rin[X.is_scz == 1], m.rin[(X.is_scz == 0) & (X.is_bd_psychosis == 0)]).pvalue,
            scz_control_depth_p=stats.mannwhitneyu(m.count_depth_million[X.is_scz == 1], m.count_depth_million[(X.is_scz == 0) & (X.is_bd_psychosis == 0)]).pvalue,
            scz_control_pmi_p=stats.mannwhitneyu(m.pmi[X.is_scz == 1], m.pmi[(X.is_scz == 0) & (X.is_bd_psychosis == 0)]).pvalue,
            scz_control_ph_p=stats.mannwhitneyu(m.ph[X.is_scz == 1], m.ph[(X.is_scz == 0) & (X.is_bd_psychosis == 0)]).pvalue,
        ))
    diag = pd.DataFrame(rows)
    for r1 in rl.REGIONS:
        for r2 in rl.REGIONS:
            if r1 != r2:
                diag.loc[diag.brain_region == r1, f"spearman_t_with_{r2}"] = stats.spearmanr(tstats[r1], tstats[r2].loc[tstats[r1].index]).statistic
    diag.to_csv(rl.OUT / "revision_B2_region_gradient_diagnostics.tsv", sep="\t", index=False)
    # cross-region concordance of FDR genes
    rows = []
    for src in rl.REGIONS:
        s = voom[(voom.brain_region == src) & (voom.contrast == "SCZ_vs_Control")].set_index("gene")
        sig = s.index[s["adj.P.Val"] < 0.10]
        for tgt in rl.REGIONS:
            if tgt == src or len(sig) == 0:
                continue
            t = voom[(voom.brain_region == tgt) & (voom.contrast == "SCZ_vs_Control")].set_index("gene").loc[sig]
            agree = np.sign(s.loc[sig].logFC) == np.sign(t.logFC)
            rows.append(dict(source_region=src, target_region=tgt, n_source_fdr10=len(sig),
                             sign_concordance=float(agree.mean()), binom_p=stats.binomtest(int(agree.sum()), len(sig), 0.5).pvalue,
                             frac_nominal_p05_in_target=float((t["P.Value"] < 0.05).mean()),
                             frac_fdr10_in_target=float((t["adj.P.Val"] < 0.10).mean()),
                             spearman_logfc=stats.spearmanr(s.loc[sig].logFC, t.logFC).statistic,
                             median_abs_logfc_source=float(s.loc[sig].logFC.abs().median()), median_abs_logfc_target=float(t.logFC.abs().median())))
    cross = pd.DataFrame(rows)
    cross.to_csv(rl.OUT / "revision_B2_cross_region_concordance.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(diag.round(4).T.to_string()); print(cross.round(4).to_string())


if __name__ == "__main__":
    main()
