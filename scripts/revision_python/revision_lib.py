"""Shared helpers for the GSE202537 striatal sensitivity analyses.

Python implementation of the limma/edgeR building blocks used by the scripts in
this folder (TMM, voom, weighted lmFit, squeezeVar/eBayes, camera, fry). The R
scripts (scripts/65_*.R) provide the reference implementation and write outputs in
the same layout; scripts/67_revision_compare_python_vs_R.py compares the two.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

import os

# Project root: scripts/revision_python/<this file> -> two levels up (override with REVISION_ROOT)
ROOT = Path(os.environ.get("REVISION_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
OUT = Path(os.environ.get("REVISION_OUT", ROOT / "results" / "tables"))   # revision_*.tsv outputs
OUT.mkdir(parents=True, exist_ok=True)
INTERMEDIATE = Path(os.environ.get("REVISION_INTERMEDIATE", ROOT / "results" / "revision_intermediate"))
INTERMEDIATE.mkdir(parents=True, exist_ok=True)

REGIONS = ["NAc", "Caudate", "Putamen"]
COVARIATES = ["age_z", "sex_male", "pmi_z", "rin_z", "ph_z", "count_depth_million_z", "tod_sin", "tod_cos"]
CONTINUOUS = ["age", "pmi", "rin", "ph", "count_depth_million"]


# ----------------------------------------------------------------------------
# Data loading (mirrors scripts/32_run_npj_hallmark_discovery.py)
# ----------------------------------------------------------------------------
def parse_gtf_gene_symbols(path: Path, needed_ids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern_gene_id = re.compile(r'gene_id "([^"]+)"')
    pattern_gene_name = re.compile(r'gene_name "([^"]+)"')
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            gene_id_match = pattern_gene_id.search(fields[8])
            gene_name_match = pattern_gene_name.search(fields[8])
            if not gene_id_match or not gene_name_match:
                continue
            gene_id = gene_id_match.group(1).split(".")[0]
            if gene_id in needed_ids:
                mapping[gene_id] = gene_name_match.group(1).upper()
    return mapping


def load_symbol_counts(cache: Path = PROCESSED / "gse202537_symbol_counts_filtered.tsv") -> pd.DataFrame:
    """Symbol-level raw counts after the manuscript's filter (>=10 counts in >=10% samples)."""
    if cache.exists():
        return pd.read_csv(cache, sep="\t", index_col=0)
    counts = pd.read_csv(RAW / "GSE202537_Read_counts_NAc_Caudate_Putamen_psychosis_mcontrols.csv.gz", index_col=0)
    counts.index = [str(i).split(".")[0] for i in counts.index]
    mapping = parse_gtf_gene_symbols(RAW / "Homo_sapiens.GRCh38.110.gtf.gz", set(counts.index))
    mapped = counts.loc[counts.index.intersection(mapping.keys())].copy()
    mapped["gene_symbol"] = [mapping[g] for g in mapped.index]
    symbol_counts = mapped.groupby("gene_symbol").sum(numeric_only=True)
    keep = (symbol_counts >= 10).sum(axis=1) >= max(3, int(np.ceil(0.10 * symbol_counts.shape[1])))
    symbol_counts = symbol_counts.loc[keep]
    symbol_counts.to_csv(cache, sep="\t")
    return symbol_counts


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(PROCESSED / "gse202537_sample_metadata.csv")
    qc = pd.read_csv(PROCESSED / "gse202537_expression_qc.csv")
    meta = meta.merge(qc[["sample_key", "count_depth_million"]], on="sample_key", how="left")
    meta["sex_male"] = (meta["gender"] == "Male").astype(float)
    meta["tod_sin"] = np.sin(2 * np.pi * meta["corrected_tod_24h"] / 24)
    meta["tod_cos"] = np.cos(2 * np.pi * meta["corrected_tod_24h"] / 24)
    meta["is_scz"] = (meta["group"] == "SCZ").astype(float)
    meta["is_bd_psychosis"] = (meta["group"] == "BD with psychosis").astype(float)
    meta["is_psychosis"] = ((meta["group"] == "SCZ") | (meta["group"] == "BD with psychosis")).astype(float)
    return meta


def region_design(meta_region: pd.DataFrame, diagnosis: str = "separate", run_fixed: bool = False,
                  extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Design matrix identical to the manuscript OLS model (region-wise standardized covariates).

    diagnosis: "separate" -> is_scz + is_bd_psychosis ; "pooled" -> is_psychosis.
    run_fixed: add sequencing-run indicators (treatment coding, first run as reference).
    extra: additional columns (e.g., RUV factors) indexed like meta_region.
    """
    d = meta_region.copy()
    for col in CONTINUOUS:
        sd = d[col].std()
        d[f"{col}_z"] = (d[col] - d[col].mean()) / sd if sd and not np.isnan(sd) else 0.0
    d["intercept"] = 1.0
    if diagnosis == "separate":
        dx_cols = ["is_scz", "is_bd_psychosis"]
    elif diagnosis == "pooled":
        dx_cols = ["is_psychosis"]
    else:
        raise ValueError(diagnosis)
    cols = ["intercept"] + dx_cols + COVARIATES
    X = d[cols].copy()
    if run_fixed:
        runs = sorted(d["sequence_id"].unique())
        for r in runs[1:]:
            X[f"run_{r}"] = (d["sequence_id"] == r).astype(float).values
    if extra is not None:
        for c in extra.columns:
            X[c] = extra.loc[d.index, c].values
    X.index = d["sample_key"].values
    return X, dx_cols


# ----------------------------------------------------------------------------
# edgeR TMM normalisation factors
# ----------------------------------------------------------------------------
def tmm_norm_factors(counts: np.ndarray, lib_size: np.ndarray | None = None, ref_column: int | None = None,
                     logratio_trim: float = 0.3, sum_trim: float = 0.05, do_weighting: bool = True,
                     a_cutoff: float = -1e10) -> np.ndarray:
    y = np.asarray(counts, dtype=float)
    if lib_size is None:
        lib_size = y.sum(axis=0)
    if ref_column is None:
        f75 = np.array([np.quantile(y[:, j] / lib_size[j], 0.75) for j in range(y.shape[1])])
        ref_column = int(np.argmin(np.abs(f75 - f75.mean())))
    f = np.ones(y.shape[1])
    for j in range(y.shape[1]):
        f[j] = _tmm_pair(y[:, j], y[:, ref_column], lib_size[j], lib_size[ref_column], logratio_trim, sum_trim,
                         do_weighting, a_cutoff)
    f = f / np.exp(np.mean(np.log(f)))
    return f


def _tmm_pair(obs, ref, nO, nR, logratio_trim, sum_trim, do_weighting, a_cutoff) -> float:
    logR = np.log2((obs / nO) / (ref / nR))
    absE = (np.log2(obs / nO) + np.log2(ref / nR)) / 2
    v = (nO - obs) / nO / obs + (nR - ref) / nR / ref
    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > a_cutoff)
    logR, absE, v = logR[fin], absE[fin], v[fin]
    if np.max(np.abs(logR)) < 1e-6:
        return 1.0
    n = len(logR)
    loL = int(np.floor(n * logratio_trim) + 1)
    hiL = n + 1 - loL
    loS = int(np.floor(n * sum_trim) + 1)
    hiS = n + 1 - loS
    rankR = stats.rankdata(logR, method="average")
    rankE = stats.rankdata(absE, method="average")
    keep = (rankR >= loL) & (rankR <= hiL) & (rankE >= loS) & (rankE <= hiS)
    if do_weighting:
        f = np.sum(logR[keep] / v[keep]) / np.sum(1 / v[keep])
    else:
        f = np.mean(logR[keep])
    if not np.isfinite(f):
        f = 0.0
    return float(2 ** f)


# ----------------------------------------------------------------------------
# limma: lmFit (optionally weighted), voom, squeezeVar, eBayes
# ----------------------------------------------------------------------------
def _lowess(x, y, frac=0.5, iters=3):
    """statsmodels-free lowess (tricube local linear, robust iterations) matching R's lowess behaviour."""
    from math import ceil
    n = len(x)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    r = int(ceil(frac * n))
    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(iters + 1):
        for i in range(n):
            # distance to r-th nearest neighbour
            dist = np.abs(xs - xs[i])
            h = np.sort(dist)[min(r, n) - 1]
            if h <= 0:
                h = 1e-12
            w = np.clip(dist / h, 0.0, 1.0)
            w = (1 - w ** 3) ** 3 * delta
            sw = np.sum(w)
            if sw <= 0:
                yest[i] = ys[i]
                continue
            xw = np.sum(w * xs) / sw
            yw = np.sum(w * ys) / sw
            sxx = np.sum(w * (xs - xw) ** 2)
            b = np.sum(w * (xs - xw) * (ys - yw)) / sxx if sxx > 0 else 0.0
            yest[i] = yw + b * (xs[i] - xw)
        resid = ys - yest
        s = np.median(np.abs(resid))
        if s <= 0:
            break
        delta = np.clip(resid / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return xs, yest


def voom(counts: np.ndarray, design: np.ndarray, lib_size: np.ndarray, span: float = 0.5):
    """limma::voom. counts: genes x samples; lib_size already multiplied by norm factors."""
    y = np.log2((counts + 0.5) / (lib_size + 1.0) * 1e6)
    coef, _, sigma, df_res, _ = lm_fit(y, design)
    amean = y.mean(axis=1)
    sx = amean + np.mean(np.log2(lib_size + 1.0)) - np.log2(1e6)
    sy = np.sqrt(sigma)
    allzero = counts.sum(axis=1) == 0
    xs, ys_fit = _lowess(sx[~allzero], sy[~allzero], frac=span)
    # approxfun with rule=2 (constant extrapolation)
    fitted_values = coef @ design.T
    fitted_cpm = 2.0 ** fitted_values
    fitted_count = 1e-6 * fitted_cpm * (lib_size + 1.0)[None, :]
    fitted_logcount = np.log2(fitted_count)
    f = np.interp(fitted_logcount, xs, ys_fit, left=ys_fit[0], right=ys_fit[-1])
    w = 1.0 / f ** 4
    return y, w, (sx, sy, xs, ys_fit)


def lm_fit(y: np.ndarray, design: np.ndarray, weights: np.ndarray | None = None):
    """Gene-wise (weighted) least squares. Returns coef, stdev_unscaled, sigma, df_resid, residuals."""
    G, n = y.shape
    p = design.shape[1]
    df_res = n - p
    coef = np.zeros((G, p))
    stdev_unscaled = np.zeros((G, p))
    sigma = np.zeros(G)
    residuals = np.zeros((G, n))
    if weights is None:
        XtX_inv = np.linalg.pinv(design.T @ design)
        coef = (XtX_inv @ design.T @ y.T).T
        residuals = y - coef @ design.T
        sigma = np.sqrt(np.sum(residuals ** 2, axis=1) / df_res)
        stdev_unscaled[:] = np.sqrt(np.diag(XtX_inv))[None, :]
    else:
        for g in range(G):
            w = weights[g]
            sw = np.sqrt(w)
            Xw = design * sw[:, None]
            yw = y[g] * sw
            XtX_inv = np.linalg.pinv(Xw.T @ Xw)
            b = XtX_inv @ Xw.T @ yw
            coef[g] = b
            r = yw - Xw @ b
            residuals[g] = (y[g] - design @ b)
            sigma[g] = np.sqrt(np.sum(r ** 2) / df_res)
            stdev_unscaled[g] = np.sqrt(np.diag(XtX_inv))
    return coef, stdev_unscaled, sigma, df_res, residuals


def trigamma_inverse(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if np.all(-dif / y < 1e-8):
            break
    return y


def fit_f_dist(x: np.ndarray, df1: float):
    """limma::fitFDist (no covariate). Returns scale (s0^2) and df2 (df.prior)."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & (x > 0)
    x = x[ok]
    z = np.log(x)
    e = z - special.digamma(df1 / 2) + np.log(df1 / 2)
    emean = np.mean(e)
    evar = np.var(e, ddof=1)
    evar = evar - special.polygamma(1, df1 / 2)
    if evar > 0:
        df2 = 2 * trigamma_inverse(np.array([evar]))[0]
        s20 = np.exp(emean + special.digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        s20 = np.exp(emean)
    return s20, df2


def squeeze_var(var: np.ndarray, df: float):
    var_prior, df_prior = fit_f_dist(var, df)
    if np.isinf(df_prior):
        var_post = np.full_like(var, var_prior)
    else:
        var_post = (df_prior * var_prior + df * var) / (df_prior + df)
    return var_post, var_prior, df_prior


def ebayes(coef, stdev_unscaled, sigma, df_res, contrast_cols, G_cap=True):
    """Moderated t for selected coefficient columns. Returns dict of DataFrames per column."""
    var_post, var_prior, df_prior = squeeze_var(sigma ** 2, df_res)
    df_total = df_res + df_prior
    df_total = min(df_total, sigma.shape[0] * df_res)
    out = {}
    for j in contrast_cols:
        t = coef[:, j] / (stdev_unscaled[:, j] * np.sqrt(var_post))
        p = 2 * stats.t.sf(np.abs(t), df_total)
        out[j] = dict(logFC=coef[:, j], t=t, p=p, df_total=df_total)
    return out, var_post, var_prior, df_prior


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    res = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.clip(ranked, 0, 1)
    res[ok] = tmp
    return res


# ----------------------------------------------------------------------------
# camera and fry (limma) on a fitted linear model
# ----------------------------------------------------------------------------
def lm_effects(y: np.ndarray, design: np.ndarray, contrast_col: int, weights: np.ndarray | None = None):
    """QR effects with the contrast column last (limma internal .lmEffects).

    Returns unscaledt (G,), U residual effects (G x df_res), df_res.
    """
    G, n = y.shape
    p = design.shape[1]
    cols = [c for c in range(p) if c != contrast_col] + [contrast_col]
    X = design[:, cols]
    effects = np.zeros((n, G))
    unscaledt = np.zeros(G)
    if weights is None:
        Q, R = np.linalg.qr(X, mode="complete")
        eff = Q.T @ y.T  # n x G
        effects = eff
        unscaledt = eff[p - 1, :]
        if R[p - 1, p - 1] < 0:
            unscaledt = -unscaledt
    else:
        sw = np.sqrt(weights)
        for g in range(G):
            Xw = X * sw[g][:, None]
            Q, R = np.linalg.qr(Xw, mode="complete")
            eff = Q.T @ (y[g] * sw[g])
            effects[:, g] = eff
            unscaledt[g] = eff[p - 1] if R[p - 1, p - 1] >= 0 else -eff[p - 1]
    U = effects[p:, :]  # df_res x G
    return unscaledt, U, n - p


def zscore_t(t: np.ndarray, df: float) -> np.ndarray:
    """Exact t -> z conversion on the log scale (limma zscoreT approx=FALSE)."""
    t = np.asarray(t, dtype=float)
    z = np.empty_like(t)
    neg = t < 0
    logp = stats.t.logcdf(t[neg], df)
    z[neg] = _qnorm_log(logp)
    logp = stats.t.logsf(t[~neg], df)
    z[~neg] = -_qnorm_log(logp)
    return z


def _qnorm_log(logp: np.ndarray) -> np.ndarray:
    # qnorm(log.p) via scipy: use ndtri on exp(logp) where safe, else asymptotic
    p = np.exp(logp)
    z = np.where(p > 1e-300, special.ndtri(np.clip(p, 1e-300, 1)), np.nan)
    small = ~np.isfinite(z) | (p <= 1e-300)
    if np.any(small):
        # asymptotic for extreme tails: z ~ -sqrt(-2*logp - log(-2*logp) - log(2*pi))? use iterative refinement
        lp = logp[small]
        zz = -np.sqrt(-2 * lp - np.log(-2 * lp) - np.log(2 * np.pi))
        for _ in range(5):
            # Newton on log Phi(z) = lp
            lphi = stats.norm.logcdf(zz)
            d = np.exp(stats.norm.logpdf(zz) - lphi)
            zz = zz - (lphi - lp) / d
        z[small] = zz
    return z


def camera(unscaledt: np.ndarray, U: np.ndarray, df_res: int, sets: dict[str, np.ndarray],
           inter_gene_cor: float | None = 0.01, allow_neg_cor: bool = False):
    """limma::camera (parametric, use.ranks=FALSE). sets: name -> boolean/index array over genes."""
    G = len(unscaledt)
    sigma2 = np.mean(U ** 2, axis=0)  # per gene
    Ustd = (U / np.sqrt(np.maximum(sigma2, 1e-8))[None, :]).T  # G x df_res
    var_post, var_prior, df_prior = squeeze_var(sigma2, df_res)
    modt = unscaledt / np.sqrt(var_post)
    df_total = min(df_res + df_prior, G * df_res)
    Stat = zscore_t(modt, df_total)
    meanStat = Stat.mean()
    varStat = Stat.var(ddof=1)
    df_camera = min(df_res, G - 2)
    rows = []
    for name, idx in sets.items():
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        m = len(idx)
        m2 = G - m
        if m < 2:
            continue
        if inter_gene_cor is not None:
            correlation = inter_gene_cor
            vif = 1 + (m - 1) * correlation
        else:
            Uset = Ustd[idx, :]
            vif = m * np.mean(np.mean(Uset, axis=0) ** 2)
            correlation = (vif - 1) / (m - 1)
        if not allow_neg_cor:
            vif = max(vif, 1.0)
        meanStatInSet = Stat[idx].mean()
        delta = G / m2 * (meanStatInSet - meanStat)
        varStatPooled = ((G - 1) * varStat - delta ** 2 * m * m2 / G) / (G - 2)
        two_sample_t = delta / np.sqrt(varStatPooled * (vif / m + 1 / m2))
        p_down = stats.t.cdf(two_sample_t, df_camera)
        p_up = stats.t.sf(two_sample_t, df_camera)
        rows.append(dict(set=name, NGenes=m, Correlation=correlation, Direction="Up" if two_sample_t > 0 else "Down",
                         two_sample_t=two_sample_t, PValue=2 * min(p_down, p_up)))
    res = pd.DataFrame(rows)
    res["FDR"] = bh(res["PValue"].values)
    return res, modt, Stat


def fry(unscaledt: np.ndarray, U: np.ndarray, df_res: int, sets: dict[str, np.ndarray]):
    """limma::fry directional test (standardize='posterior.sd')."""
    G = len(unscaledt)
    sigma2 = np.mean(U ** 2, axis=0)
    var_post, var_prior, df_prior = squeeze_var(sigma2, df_res)
    sd = np.sqrt(var_post)
    E = np.column_stack([unscaledt / sd, (U / sd[None, :]).T])  # G x (1 + df_res)
    df_use = df_res + (df_prior if np.isfinite(df_prior) else 0)
    rows = []
    for name, idx in sets.items():
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        m = len(idx)
        if m < 2:
            continue
        Eset = E[idx, :]
        meta = Eset.mean(axis=0)
        t_dir = meta[0] / np.sqrt(np.sum(meta[1:] ** 2) / df_res)
        p_dir = 2 * stats.t.sf(abs(t_dir), df_use)
        # mixed: mean of squared effects
        Fmix = np.mean(Eset[:, 0] ** 2) / np.mean(Eset[:, 1:] ** 2)
        p_mix = stats.f.sf(Fmix, 1 * 1, df_res) if False else np.nan
        rows.append(dict(set=name, NGenes=m, Direction="Up" if t_dir > 0 else "Down", t=t_dir, PValue=p_dir))
    res = pd.DataFrame(rows)
    res["FDR"] = bh(res["PValue"].values)
    return res


# ----------------------------------------------------------------------------
# Hallmark sets
# ----------------------------------------------------------------------------
def load_hallmark(expressed: pd.Index) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    defs = pd.read_csv(TABLES / "npj_figure1_hallmark_definitions.tsv", sep="\t")
    sets = {}
    for _, row in defs.iterrows():
        genes = [g for g in str(row["genes"]).split(";") if g in expressed]
        sets[row["hallmark_id"]] = genes
    return defs, sets


def gmt_hallmark(expressed: pd.Index) -> dict[str, list[str]]:
    sets = {}
    with open(RAW / "MSigDB_Hallmark_2020.gmt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            name = parts[0]
            genes = [g.upper() for g in parts[2:] if g and g.upper() in expressed]
            sets[name] = genes
    return sets
