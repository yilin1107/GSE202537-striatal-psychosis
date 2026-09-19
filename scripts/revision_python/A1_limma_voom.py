"""A1: limma-voom re-analysis of the region-specific diagnosis models (Python implementation).

Outputs (out/):
  revision_A1_voom_gene_results.tsv       gene-level voom results for 3 regions x 2 contrasts (+ OLS columns)
  revision_A1_voom_vs_ols_summary.tsv     concordance summary per region x contrast
  revision_A1_tmm_factors.tsv             TMM normalisation factors (global)
  revision_A1_voom_trend_<region>.tsv     mean-variance trend points used for the voom weights
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import revision_lib as rl


def run_region(region: str, sc: pd.DataFrame, meta: pd.DataFrame, norm_factors: pd.Series,
               run_fixed: bool = False, extra: pd.DataFrame | None = None, diagnosis: str = "separate"):
    m = meta[meta.brain_region == region].reset_index(drop=True)
    X, dx_cols = rl.region_design(m, diagnosis=diagnosis, run_fixed=run_fixed, extra=extra)
    counts = sc[X.index].to_numpy(dtype=float)
    lib = counts.sum(axis=0) * norm_factors[X.index].to_numpy()
    E, W, trend = rl.voom(counts, X.to_numpy(), lib)
    coef, su, sigma, df_res, resid = rl.lm_fit(E, X.to_numpy(), weights=W)
    cols = list(X.columns)
    contrast_idx = [cols.index(c) for c in dx_cols]
    eb, var_post, var_prior, df_prior = rl.ebayes(coef, su, sigma, df_res, contrast_idx)
    out = []
    for c, j in zip(dx_cols, contrast_idx):
        contrast = {"is_scz": "SCZ_vs_Control", "is_bd_psychosis": "BD_psychosis_vs_Control", "is_psychosis": "Psychosis_vs_Control"}[c]
        df = pd.DataFrame({
            "gene": sc.index,
            "brain_region": region,
            "contrast": contrast,
            "n_samples": X.shape[0],
            "AveExpr": E.mean(axis=1),
            "logFC": eb[j]["logFC"],
            "t": eb[j]["t"],
            "P.Value": eb[j]["p"],
        })
        df["adj.P.Val"] = rl.bh(df["P.Value"].values)
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    info = dict(region=region, n=X.shape[0], p=X.shape[1], df_res=df_res, df_prior=df_prior, var_prior=var_prior)
    return res, info, trend, dict(E=E, W=W, X=X, coef=coef, su=su, sigma=sigma, df_res=df_res, resid=resid, var_post=var_post)


def concordance(voom_res: pd.DataFrame, ols: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, contrast), v in voom_res.groupby(["brain_region", "contrast"]):
        o = ols[(ols.brain_region == region) & (ols.contrast == contrast)].set_index("gene").loc[v.gene]
        v = v.set_index("gene")
        sig_o = set(o.index[o.fdr < 0.10])
        sig_v = set(v.index[v["adj.P.Val"] < 0.10])
        both = sig_o & sig_v
        rows.append({
            "brain_region": region, "contrast": contrast,
            "n_genes": len(v),
            "ols_fdr10": len(sig_o), "voom_fdr10": len(sig_v), "overlap_fdr10": len(both),
            "jaccard_fdr10": len(both) / len(sig_o | sig_v) if (sig_o | sig_v) else np.nan,
            "ols_fdr05": int((o.fdr < 0.05).sum()), "voom_fdr05": int((v["adj.P.Val"] < 0.05).sum()),
            "spearman_logfc": stats.spearmanr(o.log2fc, v.logFC).statistic,
            "pearson_logfc": stats.pearsonr(o.log2fc, v.logFC).statistic,
            "spearman_t": stats.spearmanr(o.t_value, v.t).statistic,
            "sign_concordance_all": float(np.mean(np.sign(o.log2fc) == np.sign(v.logFC))),
            "sign_concordance_ols_fdr10": float(np.mean(np.sign(o.loc[list(sig_o)].log2fc) == np.sign(v.loc[list(sig_o)].logFC))) if sig_o else np.nan,
            "min_voom_fdr": float(v["adj.P.Val"].min()),
            "voom_top_gene": v["adj.P.Val"].idxmin(),
        })
    return pd.DataFrame(rows)


def main():
    sc = rl.load_symbol_counts()
    meta = rl.load_metadata()
    ols = pd.read_csv(rl.TABLES / "npj_figure1_gene_ma_results.tsv", sep="\t")
    nf = pd.Series(rl.tmm_norm_factors(sc.to_numpy(dtype=float)), index=sc.columns)
    nf.rename("norm_factor").to_frame().assign(lib_size=sc.sum(axis=0)).to_csv(rl.OUT / "revision_A1_tmm_factors.tsv", sep="\t")
    print("TMM factors: min %.3f max %.3f" % (nf.min(), nf.max()))
    all_res, infos = [], []
    for region in rl.REGIONS:
        res, info, trend, fit = run_region(region, sc, meta, nf)
        sx, sy, xs, ys = trend
        pd.DataFrame({"sx": sx, "sy": sy}).to_csv(rl.OUT / f"revision_A1_voom_trend_{region}.tsv", sep="\t", index=False)
        all_res.append(res)
        infos.append(info)
        print(region, info)
        np.save(rl.INTERMEDIATE / f"A1_fit_{region}_E.npy", fit["E"]); np.save(rl.INTERMEDIATE / f"A1_fit_{region}_W.npy", fit["W"])
        fit["X"].to_csv(rl.INTERMEDIATE / f"A1_fit_{region}_design.tsv", sep="\t")
    res = pd.concat(all_res, ignore_index=True)
    # attach OLS columns
    o = ols.set_index(["gene", "brain_region", "contrast"])[["log2fc", "t_value", "p_value", "fdr"]]
    o.columns = ["ols_log2fc", "ols_t", "ols_p", "ols_fdr"]
    res = res.join(o, on=["gene", "brain_region", "contrast"])
    res.to_csv(rl.OUT / "revision_A1_voom_gene_results.tsv", sep="\t", index=False)
    summ = concordance(res, ols)
    summ.to_csv(rl.OUT / "revision_A1_voom_vs_ols_summary.tsv", sep="\t", index=False)
    pd.DataFrame(infos).to_csv(rl.OUT / "revision_A1_model_info.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    main()
