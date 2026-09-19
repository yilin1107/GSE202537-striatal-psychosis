"""A2: sequencing-run sensitivity analyses on the voom framework.

Variants (all region-specific, same covariates as the primary model):
  primary        voom (A1), no run term
  run_fixed      + sequencing-run indicators (fixed effects)
  dupcor         limma duplicateCorrelation-style random block for run (consensus correlation, GLS fit)
  ruvr_k1..k3    RUVr-style residual factors (SVD of first-pass residuals) added to the design

Outputs (out/):
  revision_A2_run_sensitivity_gene_results.tsv   long table (variant x region x contrast x gene)
  revision_A2_run_sensitivity_summary.tsv        concordance with the primary model
  revision_A2_run_sensitivity_hallmark_camera.tsv camera (0.01) FDR per Hallmark set under each variant
  revision_A2_run_structure.tsv                  run x diagnosis counts per region
  revision_A2_downstream_stability.tsv           signature scores / NNLS OPC weights with run fixed effect
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import revision_lib as rl
from A1_limma_voom import run_region


# ---------------------------------------------------------------------------
# duplicateCorrelation-style consensus correlation + GLS fit
# ---------------------------------------------------------------------------
def block_matrix(block: np.ndarray) -> np.ndarray:
    levels = {b: i for i, b in enumerate(sorted(set(block)))}
    Z = np.zeros((len(block), len(levels)))
    for i, b in enumerate(block):
        Z[i, levels[b]] = 1.0
    return Z


def reml_rho_per_gene(y: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray | None,
                      grid=np.linspace(0.0, 0.9, 46)) -> np.ndarray:
    """Per-gene REML estimate of the within-block correlation (grid search)."""
    n, p = X.shape
    G = y.shape[0]
    ZZt = Z @ Z.T
    ll = np.full((G, len(grid)), -np.inf)
    for k, rho in enumerate(grid):
        V = (1 - rho) * np.eye(n) + rho * ZZt
        if W is None:
            L = np.linalg.cholesky(V)
            Xt = np.linalg.solve(L, X)
            Yt = np.linalg.solve(L, y.T)  # n x G
            XtX = Xt.T @ Xt
            beta = np.linalg.solve(XtX, Xt.T @ Yt)
            R = Yt - Xt @ beta
            rss = np.sum(R ** 2, axis=0)
            logdetV = 2 * np.sum(np.log(np.diag(L)))
            logdetXX = np.linalg.slogdet(XtX)[1]
            ll[:, k] = -0.5 * (logdetV + logdetXX + (n - p) * np.log(rss / (n - p)))
        else:
            for g in range(G):
                w = W[g]
                # weights: y_w = sqrt(w) y ; V_w = D V D with D = diag(1/sqrt(w))  -> equivalently transform
                D = 1.0 / np.sqrt(w)
                Vw = V * np.outer(D, D)
                L = np.linalg.cholesky(Vw)
                Xt = np.linalg.solve(L, X)
                yt = np.linalg.solve(L, y[g])
                XtX = Xt.T @ Xt
                beta = np.linalg.solve(XtX, Xt.T @ yt)
                r = yt - Xt @ beta
                rss = float(r @ r)
                logdetV = 2 * np.sum(np.log(np.diag(L)))
                logdetXX = np.linalg.slogdet(XtX)[1]
                ll[g, k] = -0.5 * (logdetV + logdetXX + (n - p) * np.log(rss / (n - p)))
    best = grid[np.argmax(ll, axis=1)]
    return best


def consensus_correlation(rho: np.ndarray, trim: float = 0.15) -> float:
    z = np.arctanh(np.clip(rho, -0.999, 0.999))
    return float(np.tanh(stats.trim_mean(z, trim)))


def gls_fit(y: np.ndarray, X: np.ndarray, Z: np.ndarray, rho: float, W: np.ndarray | None):
    """limma gls.series-style fit with block correlation rho (and optional gene-wise weights)."""
    n, p = X.shape
    G = y.shape[0]
    V = (1 - rho) * np.eye(n) + rho * (Z @ Z.T)
    coef = np.zeros((G, p)); su = np.zeros((G, p)); sigma = np.zeros(G)
    if W is None:
        L = np.linalg.cholesky(V)
        Xt = np.linalg.solve(L, X)
        Yt = np.linalg.solve(L, y.T)
        XtX_inv = np.linalg.pinv(Xt.T @ Xt)
        coef = (XtX_inv @ Xt.T @ Yt).T
        R = Yt - Xt @ coef.T
        sigma = np.sqrt(np.sum(R ** 2, axis=0) / (n - p))
        su[:] = np.sqrt(np.diag(XtX_inv))[None, :]
    else:
        for g in range(G):
            D = 1.0 / np.sqrt(W[g])
            L = np.linalg.cholesky(V * np.outer(D, D))
            Xt = np.linalg.solve(L, X)
            yt = np.linalg.solve(L, y[g])
            XtX_inv = np.linalg.pinv(Xt.T @ Xt)
            b = XtX_inv @ Xt.T @ yt
            coef[g] = b
            r = yt - Xt @ b
            sigma[g] = np.sqrt(float(r @ r) / (n - p))
            su[g] = np.sqrt(np.diag(XtX_inv))
    return coef, su, sigma, n - p


def results_frame(region, X, dx_cols, coef, su, sigma, df_res, genes, E):
    cols = list(X.columns)
    contrast_idx = [cols.index(c) for c in dx_cols]
    eb, var_post, var_prior, df_prior = rl.ebayes(coef, su, sigma, df_res, contrast_idx)
    out = []
    for c, j in zip(dx_cols, contrast_idx):
        contrast = {"is_scz": "SCZ_vs_Control", "is_bd_psychosis": "BD_psychosis_vs_Control"}[c]
        df = pd.DataFrame({"gene": genes, "brain_region": region, "contrast": contrast, "AveExpr": E.mean(axis=1),
                           "logFC": eb[j]["logFC"], "t": eb[j]["t"], "P.Value": eb[j]["p"]})
        df["adj.P.Val"] = rl.bh(df["P.Value"].values)
        out.append(df)
    return pd.concat(out, ignore_index=True), dict(df_res=df_res, df_prior=df_prior)


def camera_all_sets(E, W, X, sets):
    cols = list(X.columns)
    rows = []
    for col, contrast in [("is_scz", "SCZ_vs_Control"), ("is_bd_psychosis", "BD_psychosis_vs_Control")]:
        j = cols.index(col)
        unscaledt, U, df_res = rl.lm_effects(E, X.to_numpy(), j, weights=W)
        cam, _, _ = rl.camera(unscaledt, U, df_res, sets, inter_gene_cor=0.01)
        cam.insert(0, "contrast", contrast)
        rows.append(cam)
    return pd.concat(rows, ignore_index=True)


def main():
    sc = rl.load_symbol_counts()
    genes = sc.index
    meta = rl.load_metadata()
    nf = pd.read_csv(rl.OUT / "revision_A1_tmm_factors.tsv", sep="\t", index_col=0)["norm_factor"]
    primary = pd.read_csv(rl.OUT / "revision_A1_voom_gene_results.tsv", sep="\t")
    # Hallmark sets
    defs = pd.read_csv(rl.TABLES / "npj_figure1_hallmark_definitions.tsv", sep="\t")
    gene_pos = {g: i for i, g in enumerate(genes)}
    gmt = {}
    with open(rl.RAW / "MSigDB_Hallmark_2020.gmt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gmt[parts[0].strip()] = sorted({g.strip().upper() for g in parts[2:] if g.strip()})
    sets = {row["hallmark_id"]: np.array([gene_pos[g] for g in gmt[row["hallmark"]] if g in gene_pos]) for _, row in defs.iterrows()}

    # run structure table
    struct = meta.groupby(["brain_region", "sequence_id", "group"]).size().unstack(fill_value=0).reset_index()
    struct.to_csv(rl.OUT / "revision_A2_run_structure.tsv", sep="\t", index=False)

    all_gene, all_cam, infos = [], [], []
    for region in rl.REGIONS:
        m = meta[meta.brain_region == region].reset_index(drop=True)
        # --- primary fit objects (from A1)
        E0 = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_E.npy"); W0 = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_W.npy")
        X0 = pd.read_csv(rl.INTERMEDIATE / f"A1_fit_{region}_design.tsv", sep="\t", index_col=0)
        cam0 = camera_all_sets(E0, W0, X0, sets); cam0.insert(0, "variant", "primary"); cam0.insert(0, "brain_region", region)
        all_cam.append(cam0)

        # --- run fixed effect
        res, info, trend, fit = run_region(region, sc, meta, nf, run_fixed=True)
        res.insert(0, "variant", "run_fixed"); all_gene.append(res)
        infos.append(dict(variant="run_fixed", **info))
        cam = camera_all_sets(fit["E"], fit["W"], fit["X"], sets); cam.insert(0, "variant", "run_fixed"); cam.insert(0, "brain_region", region)
        all_cam.append(cam)
        print(region, "run_fixed", info)

        # --- duplicateCorrelation-style block random effect (voom weights from primary fit)
        Z = block_matrix(m.set_index("sample_key").loc[X0.index, "sequence_id"].values)
        rho_g = reml_rho_per_gene(E0, X0.to_numpy(), Z, W0)
        rho = consensus_correlation(rho_g)
        coef, su, sigma, df_res = gls_fit(E0, X0.to_numpy(), Z, rho, W0)
        res, info = results_frame(region, X0, ["is_scz", "is_bd_psychosis"], coef, su, sigma, df_res, genes, E0)
        res.insert(0, "variant", "dupcor"); all_gene.append(res)
        infos.append(dict(variant="dupcor", region=region, consensus_correlation=rho, median_gene_rho=float(np.median(rho_g)), **info))
        print(region, "dupcor consensus rho=%.3f" % rho, info)
        # camera for dupcor: approximate by pre-whitening (transform E and design with block correlation)
        L = np.linalg.cholesky((1 - rho) * np.eye(len(Z)) + rho * (Z @ Z.T))
        Et = (np.linalg.solve(L, E0.T)).T
        Xt = pd.DataFrame(np.linalg.solve(L, X0.to_numpy()), index=X0.index, columns=X0.columns)
        cam = camera_all_sets(Et, W0, Xt, sets); cam.insert(0, "variant", "dupcor"); cam.insert(0, "brain_region", region)
        all_cam.append(cam)

        # --- RUVr-style residual factors
        # residuals of primary weighted fit
        coef0, su0, sigma0, dfr0, resid0 = rl.lm_fit(E0, X0.to_numpy(), weights=W0)
        Rc = resid0 - resid0.mean(axis=1, keepdims=True)
        u, s, vt = np.linalg.svd(Rc, full_matrices=False)
        for k in (1, 2, 3):
            extra = pd.DataFrame(vt[:k].T, index=m.index, columns=[f"ruv{i+1}" for i in range(k)])
            res, info, trend, fit = run_region(region, sc, meta, nf, extra=extra)
            res.insert(0, "variant", f"ruvr_k{k}"); all_gene.append(res)
            infos.append(dict(variant=f"ruvr_k{k}", **info))
            cam = camera_all_sets(fit["E"], fit["W"], fit["X"], sets); cam.insert(0, "variant", f"ruvr_k{k}"); cam.insert(0, "brain_region", region)
            all_cam.append(cam)
            print(region, f"ruvr_k{k}", info)

    gene_res = pd.concat(all_gene, ignore_index=True)
    gene_res.to_csv(rl.OUT / "revision_A2_run_sensitivity_gene_results.tsv", sep="\t", index=False)
    cam_res = pd.concat(all_cam, ignore_index=True)
    cam_res.to_csv(rl.OUT / "revision_A2_run_sensitivity_hallmark_camera.tsv", sep="\t", index=False)
    pd.DataFrame(infos).to_csv(rl.OUT / "revision_A2_model_info.tsv", sep="\t", index=False)

    # concordance with primary
    rows = []
    prim = primary.set_index(["brain_region", "contrast", "gene"])
    for (variant, region, contrast), v in gene_res.groupby(["variant", "brain_region", "contrast"]):
        v = v.set_index("gene")
        p0 = prim.loc[(region, contrast)].loc[v.index]
        sig0 = set(p0.index[p0["adj.P.Val"] < 0.10]); sig1 = set(v.index[v["adj.P.Val"] < 0.10])
        cam_v = cam_res[(cam_res.variant == variant) & (cam_res.brain_region == region) & (cam_res.contrast == contrast)]
        cam_0 = cam_res[(cam_res.variant == "primary") & (cam_res.brain_region == region) & (cam_res.contrast == contrast)]
        rows.append({
            "variant": variant, "brain_region": region, "contrast": contrast,
            "primary_fdr10": len(sig0), "variant_fdr10": len(sig1), "overlap": len(sig0 & sig1),
            "spearman_t_vs_primary": stats.spearmanr(p0.t, v.t).statistic,
            "pearson_logfc_vs_primary": stats.pearsonr(p0.logFC, v.logFC).statistic,
            "sign_concordance_primary_fdr10": float(np.mean(np.sign(p0.loc[list(sig0)].logFC) == np.sign(v.loc[list(sig0)].logFC))) if sig0 else np.nan,
            "primary_fdr10_genes_nominal_p05_in_variant": float(np.mean(v.loc[list(sig0)]["P.Value"] < 0.05)) if sig0 else np.nan,
            "camera_fdr10_primary": int((cam_0.FDR < 0.1).sum()), "camera_fdr10_variant": int((cam_v.FDR < 0.1).sum()),
            "camera_fdr10_overlap": len(set(cam_0.set[cam_0.FDR < 0.1]) & set(cam_v.set[cam_v.FDR < 0.1])),
            "spearman_camera_t_vs_primary": stats.spearmanr(cam_0.set_index("set").loc[cam_v.set].two_sample_t, cam_v.two_sample_t).statistic,
        })
    summ = pd.DataFrame(rows).sort_values(["contrast", "brain_region", "variant"])
    summ.to_csv(rl.OUT / "revision_A2_run_sensitivity_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    main()
