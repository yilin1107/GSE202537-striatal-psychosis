"""A6: alternative estimation of brain-resident composition (putamen OPC finding).

Part 1  Linear-scale NNLS with the same HPA/Siletti marker reference (the manuscript used log2 scale),
        plus a 'w/o OPC-vs-COP split' check; SCZ-vs-control arcsine OLS as in scripts/47 (with and without run).
Part 2  Profile-based simulation benchmark: mixtures synthesised from HPA cluster profiles with known
        proportions (Dirichlet around striatal composition, OPC 1-10 %), Poisson-sampled at bulk depth,
        deconvolved with log-NNLS and linear-NNLS; recovery of OPC and power to detect a 40 % OPC reduction.
Part 3  Reference-perturbation check: markers re-derived after leaving out each contributing cluster.

Outputs (out/): revision_A6_nnls_weights_long.tsv, revision_A6_nnls_tests.tsv,
                revision_A6_simulation_recovery.tsv, revision_A6_simulation_power.tsv, revision_A6_leave_cluster_out.tsv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import revision_lib as rl

BRAIN_CLASSES = {
    "MSN": ["medium spiny neuron", "eccentric medium spiny neuron"],
    "Interneuron": ["CGE interneuron", "MGE interneuron", "LAMP5-LHX6 and Chandelier"],
    "Astrocyte": ["astrocyte"],
    "Oligodendrocyte": ["oligodendrocyte"],
    "OPC": ["oligodendrocyte precursor cell", "committed oligodendrocyte precursor"],
    "Microglia": ["central nervous system macrophage"],
    "Vascular": ["endothelial cell", "pericyte", "vascular associated smooth muscle cell"],
}
CLASSES = list(BRAIN_CLASSES)


def load_hpa():
    hpa = pd.read_csv(rl.RAW / "rna_single_nuclei_cluster_type.tsv.zip", sep="\t", compression="zip")
    hpa["Gene name"] = hpa["Gene name"].astype(str).str.upper()
    expr = hpa.pivot_table(index="Gene name", columns="Cluster type", values="nCPM", aggfunc="mean").fillna(0.0)
    return expr


def build_reference(expr: pd.DataFrame, classes: dict[str, list[str]], top: int = 80):
    ref = pd.DataFrame(index=expr.index)
    rows = []
    all_clusters = list(expr.columns)
    for cls, clusters in classes.items():
        present = [c for c in clusters if c in expr.columns]
        ref[cls] = expr[present].mean(axis=1)
        other = [c for c in all_clusters if c not in present]
        spec = np.log2((ref[cls] + 1.0) / (expr[other].max(axis=1) + 1.0))
        tab = pd.DataFrame({"gene": expr.index, "cls": cls, "group_nCPM": ref[cls], "specificity": spec})
        tab = tab[tab.group_nCPM >= 1.0].sort_values(["specificity", "group_nCPM"], ascending=[False, False]).head(top)
        rows.append(tab)
    return ref[list(classes)], pd.concat(rows, ignore_index=True)


def nnls_deconvolve(ref: pd.DataFrame, bulk: pd.DataFrame, markers: list[str], scale: str) -> pd.DataFrame:
    genes = [g for g in markers if g in bulk.index and g in ref.index]
    R = ref.loc[genes].to_numpy(dtype=float)
    Bm = bulk.loc[genes].to_numpy(dtype=float)
    if scale == "log":
        R = np.log2(R + 1.0); Bm = np.log2(Bm + 1.0)
    W = np.zeros((len(ref.columns), Bm.shape[1]))
    for j in range(Bm.shape[1]):
        coef, _ = optimize.nnls(R, Bm[:, j])
        W[:, j] = coef / coef.sum() if coef.sum() > 0 else 1.0 / len(coef)
    return pd.DataFrame(W, index=ref.columns, columns=bulk.columns)


def arcsine_tests(weights: pd.DataFrame, meta: pd.DataFrame, label: str, run_fixed: bool = False) -> pd.DataFrame:
    rows = []
    for region in rl.REGIONS:
        m = meta[(meta.brain_region == region) & (meta.group.isin(["Control", "SCZ"]))].reset_index(drop=True)
        X, _ = rl.region_design(m, diagnosis="pooled", run_fixed=run_fixed)  # is_psychosis == SCZ here
        for cls in weights.index:
            y = np.arcsin(np.sqrt(np.clip(weights.loc[cls, X.index].to_numpy(dtype=float), 0, 1)))
            if np.all(y == y[0]):
                rows.append(dict(method=label, brain_region=region, brain_cell_class=cls, beta=np.nan, p=np.nan))
                continue
            Xm = X.to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            r = y - Xm @ beta; df = len(y) - Xm.shape[1]
            cov = np.linalg.pinv(Xm.T @ Xm) * float(r @ r) / df
            t = beta[1] / np.sqrt(cov[1, 1])
            ctrl = weights.loc[cls, X.index[X.is_psychosis == 0]].mean(); scz = weights.loc[cls, X.index[X.is_psychosis == 1]].mean()
            rows.append(dict(method=label, brain_region=region, brain_cell_class=cls, control_mean=ctrl, scz_mean=scz,
                             mean_difference=scz - ctrl, beta=beta[1], se=np.sqrt(cov[1, 1]),
                             ci_low=beta[1] - stats.t.ppf(0.975, df) * np.sqrt(cov[1, 1]),
                             ci_high=beta[1] + stats.t.ppf(0.975, df) * np.sqrt(cov[1, 1]),
                             t=t, p=2 * stats.t.sf(abs(t), df)))
    res = pd.DataFrame(rows)
    res["fdr_region"] = res.groupby("brain_region")["p"].transform(lambda x: rl.bh(x.values))
    res["fdr_global"] = rl.bh(res["p"].values)
    return res


def simulate(expr: pd.DataFrame, ref: pd.DataFrame, markers: list[str], rng: np.random.Generator,
             n_mix: int = 300, depth: int = 30_000_000):
    """Mixtures from cluster-level profiles (cluster chosen at random within class) with known proportions."""
    base = np.array([0.45, 0.03, 0.15, 0.24, 0.04, 0.05, 0.04])  # MSN, IN, Astro, Oligo, OPC, Micro, Vasc
    genes = expr.index
    truths, props_log, props_lin = [], [], []
    for i in range(n_mix):
        p = rng.dirichlet(base * 40)
        # OPC spread over 0.5-10 %
        p[4] = rng.uniform(0.005, 0.10); p = p / p.sum()
        profile = np.zeros(len(genes))
        for k, cls in enumerate(CLASSES):
            clusters = [c for c in BRAIN_CLASSES[cls] if c in expr.columns]
            c = clusters[rng.integers(len(clusters))]
            v = expr[c].to_numpy(dtype=float)
            profile += p[k] * v / v.sum()
        counts = rng.poisson(profile / profile.sum() * depth)
        cpm = counts / counts.sum() * 1e6
        b = pd.DataFrame({"mix": cpm}, index=genes)
        props_log.append(nnls_deconvolve(ref, b, markers, "log")["mix"].to_numpy())
        props_lin.append(nnls_deconvolve(ref, b, markers, "linear")["mix"].to_numpy())
        truths.append(p)
    T = np.array(truths); L = np.array(props_log); N = np.array(props_lin)
    rows = []
    for k, cls in enumerate(CLASSES):
        for label, M in [("log_nnls", L), ("linear_nnls", N)]:
            r = stats.pearsonr(T[:, k], M[:, k]).statistic if np.std(M[:, k]) > 0 else np.nan
            rows.append(dict(brain_cell_class=cls, method=label, truth_mean=T[:, k].mean(), est_mean=M[:, k].mean(),
                             pearson_r=r, spearman_r=stats.spearmanr(T[:, k], M[:, k]).statistic if np.std(M[:, k]) > 0 else np.nan,
                             rmse=float(np.sqrt(np.mean((T[:, k] - M[:, k]) ** 2))), bias=float(np.mean(M[:, k] - T[:, k])),
                             slope=float(np.polyfit(T[:, k], M[:, k], 1)[0]) if np.std(M[:, k]) > 0 else np.nan))
    return pd.DataFrame(rows), T, L, N


def simulate_power(expr, ref, markers, rng, n_rep: int = 200, n_ctrl: int = 36, n_scz: int = 28,
                   opc_ctrl: float = 0.027, opc_scz: float = 0.0155, depth: int = 30_000_000):
    base = np.array([0.45, 0.03, 0.15, 0.24, 0.04, 0.05, 0.04])
    genes = expr.index
    def one_group(n, opc):
        est_log, est_lin = [], []
        for i in range(n):
            p = rng.dirichlet(base * 40); p[4] = max(1e-4, rng.normal(opc, opc * 0.3)); p = p / p.sum()
            profile = np.zeros(len(genes))
            for k, cls in enumerate(CLASSES):
                clusters = [c for c in BRAIN_CLASSES[cls] if c in expr.columns]
                v = expr[clusters[rng.integers(len(clusters))]].to_numpy(dtype=float)
                profile += p[k] * v / v.sum()
            counts = rng.poisson(profile / profile.sum() * depth)
            b = pd.DataFrame({"mix": counts / counts.sum() * 1e6}, index=genes)
            est_log.append(nnls_deconvolve(ref, b, markers, "log")["mix"].to_numpy()[4])
            est_lin.append(nnls_deconvolve(ref, b, markers, "linear")["mix"].to_numpy()[4])
        return np.array(est_log), np.array(est_lin)
    hits = {"log_nnls": 0, "linear_nnls": 0}; diffs = {"log_nnls": [], "linear_nnls": []}
    for r in range(n_rep):
        cl, cn = one_group(n_ctrl, opc_ctrl); sl, sn = one_group(n_scz, opc_scz)
        for label, c, s in [("log_nnls", cl, sl), ("linear_nnls", cn, sn)]:
            p = stats.ttest_ind(np.arcsin(np.sqrt(s)), np.arcsin(np.sqrt(c))).pvalue
            hits[label] += p < 0.05; diffs[label].append(s.mean() - c.mean())
    return pd.DataFrame([dict(method=k, n_rep=n_rep, power_alpha05=hits[k] / n_rep, mean_est_difference=np.mean(diffs[k]),
                              true_difference=opc_scz - opc_ctrl) for k in hits])


def main():
    rng = np.random.default_rng(20260919)
    meta = rl.load_metadata()
    expr = load_hpa()
    ref, markers = build_reference(expr, BRAIN_CLASSES)
    src = pd.read_csv(rl.TABLES / "npj_figure4_hpa_brain_resident_source_markers.tsv", sep="\t")
    assert set(markers.gene) == set(src.gene), "marker set differs from manuscript"
    bulk = pd.read_csv(rl.PROCESSED / "gse202537_gene_cpm_for_iobr.tsv", sep="\t", index_col=0)
    bulk.index = bulk.index.astype(str).str.upper(); bulk = bulk.groupby(bulk.index).mean()
    marker_genes = sorted(set(markers.gene) & set(bulk.index))
    print("markers available:", len(marker_genes))
    w_log = nnls_deconvolve(ref, bulk, marker_genes, "log")
    w_lin = nnls_deconvolve(ref, bulk, marker_genes, "linear")
    # validate log-scale reproduction
    orig = pd.read_csv(rl.TABLES / "npj_figure4_hpa_brain_resident_nnls_weights_wide.tsv", sep="\t", index_col=0)
    common = [c for c in orig.columns if c in w_log.columns]
    print("max |log-NNLS weight diff| vs manuscript:", float((orig.loc[w_log.index, common] - w_log[common]).abs().max().max()))
    long = pd.concat([w_log.reset_index(names="brain_cell_class").melt(id_vars="brain_cell_class", var_name="sample_key", value_name="weight").assign(method="log_nnls"),
                      w_lin.reset_index(names="brain_cell_class").melt(id_vars="brain_cell_class", var_name="sample_key", value_name="weight").assign(method="linear_nnls")])
    long = long.merge(meta[["sample_key", "brain_region", "group", "patients_id"]], on="sample_key")
    long.to_csv(rl.OUT / "revision_A6_nnls_weights_long.tsv", sep="\t", index=False)
    tests = pd.concat([arcsine_tests(w_log, meta, "log_nnls"), arcsine_tests(w_lin, meta, "linear_nnls"),
                       arcsine_tests(w_log, meta, "log_nnls_run_fixed", run_fixed=True), arcsine_tests(w_lin, meta, "linear_nnls_run_fixed", run_fixed=True)])
    tests.to_csv(rl.OUT / "revision_A6_nnls_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(tests[tests.brain_cell_class.isin(["OPC", "Oligodendrocyte"])].round(4).to_string())
    print("mean weights by method (all samples):"); print(pd.concat([w_log.mean(axis=1).rename("log"), w_lin.mean(axis=1).rename("linear")], axis=1).round(3))
    # simulations
    rec, T, L, N = simulate(expr, ref, marker_genes, rng)
    rec.to_csv(rl.OUT / "revision_A6_simulation_recovery.tsv", sep="\t", index=False)
    print(rec.round(3).to_string())
    pw = simulate_power(expr, ref, marker_genes, rng)
    pw.to_csv(rl.OUT / "revision_A6_simulation_power.tsv", sep="\t", index=False)
    print(pw.round(3).to_string())
    # leave-one-cluster-out reference perturbation for OPC (and all classes)
    rows = []
    for cls, clusters in BRAIN_CLASSES.items():
        if len(clusters) < 2:
            continue
        for drop in clusters:
            classes2 = {k: ([c for c in v if c != drop] if k == cls else v) for k, v in BRAIN_CLASSES.items()}
            ref2, markers2 = build_reference(expr, classes2)
            mg2 = sorted(set(markers2.gene) & set(bulk.index))
            for scale in ["log", "linear"]:
                w2 = nnls_deconvolve(ref2, bulk, mg2, scale)
                t2 = arcsine_tests(w2, meta, f"{scale}_nnls_drop_{drop}")
                t2 = t2[(t2.brain_region == "Putamen") & (t2.brain_cell_class.isin(["OPC", "Oligodendrocyte"]))]
                t2["perturbed_class"] = cls; t2["dropped_cluster"] = drop; t2["scale"] = scale
                rows.append(t2)
    lco = pd.concat(rows, ignore_index=True)
    lco.to_csv(rl.OUT / "revision_A6_leave_cluster_out.tsv", sep="\t", index=False)
    print(lco[["scale", "perturbed_class", "dropped_cluster", "brain_cell_class", "beta", "p", "fdr_region"]].round(4).to_string())


if __name__ == "__main__":
    main()
