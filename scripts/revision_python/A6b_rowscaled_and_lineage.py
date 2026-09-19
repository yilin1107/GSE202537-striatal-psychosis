"""A6b: row-scaled linear NNLS (per-gene scaling by the reference row mean), cross-method comparison tables,
and oligodendrocyte-lineage (OPC + oligodendrocyte) tests for all deconvolution methods.

Outputs (out/):
  revision_A6_rowscaled_nnls_weights_long.tsv
  revision_A6_nnls_tests.tsv                 (rows for linear_nnls_rowscaled[_run_fixed] appended)
  revision_A6_ol_lineage_tests.tsv           (log_nnls, linear_nnls_rowscaled, cibersort_hpa, music_tran_nac)
  revision_A6_method_concordance.tsv         (Spearman across methods per class, 215 samples)
  revision_A6_method_mean_composition.tsv    (mean estimated composition per method)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import revision_lib as rl
import A6_opc_alternative as A6


def rowscaled_nnls(ref: pd.DataFrame, bulk: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    R = ref.loc[genes].to_numpy(float); B = bulk.loc[genes].to_numpy(float)
    s = R.mean(axis=1, keepdims=True) + 1e-9
    R = R / s; B = B / s
    W = np.zeros((R.shape[1], B.shape[1]))
    for j in range(B.shape[1]):
        c, _ = optimize.nnls(R, B[:, j]); W[:, j] = c / c.sum() if c.sum() > 0 else 1 / len(c)
    return pd.DataFrame(W, index=ref.columns, columns=bulk.columns)


def main():
    meta = rl.load_metadata()
    expr = A6.load_hpa(); ref, markers = A6.build_reference(expr, A6.BRAIN_CLASSES)
    bulk = pd.read_csv(rl.PROCESSED / "gse202537_gene_cpm_for_iobr.tsv", sep="\t", index_col=0)
    bulk.index = bulk.index.astype(str).str.upper(); bulk = bulk.groupby(bulk.index).mean()
    mg = sorted(set(markers.gene) & set(bulk.index))
    w_rs = rowscaled_nnls(ref, bulk, mg)
    long = w_rs.reset_index(names="brain_cell_class").melt(id_vars="brain_cell_class", var_name="sample_key", value_name="weight").assign(method="linear_nnls_rowscaled")
    long = long.merge(meta[["sample_key", "brain_region", "group", "patients_id"]], on="sample_key")
    long.to_csv(rl.OUT / "revision_A6_rowscaled_nnls_weights_long.tsv", sep="\t", index=False)
    tests = pd.read_csv(rl.OUT / "revision_A6_nnls_tests.tsv", sep="\t")
    tests = tests[~tests.method.str.startswith("linear_nnls_rowscaled")]
    t_rs = A6.arcsine_tests(w_rs, meta, "linear_nnls_rowscaled")
    t_rs_run = A6.arcsine_tests(w_rs, meta, "linear_nnls_rowscaled_run_fixed", run_fixed=True)
    tests = pd.concat([tests, t_rs, t_rs_run], ignore_index=True)
    tests.to_csv(rl.OUT / "revision_A6_nnls_tests.tsv", sep="\t", index=False)

    # all methods as class x sample matrices
    w_all = pd.read_csv(rl.OUT / "revision_A6_nnls_weights_long.tsv", sep="\t")
    lognnls = w_all[w_all.method == "log_nnls"].pivot(index="brain_cell_class", columns="sample_key", values="weight")
    mus = pd.read_csv(rl.TABLES / "revision_R_A6_music_tran_nac_proportions.tsv", sep="\t", index_col=0).T
    cib = pd.read_csv(rl.TABLES / "revision_R_A6_cibersort_hpa_proportions.tsv", sep="\t", index_col=0)
    cib = cib[["MSN", "Interneuron", "Astrocyte", "Oligodendrocyte", "OPC", "Microglia", "Vascular"]].T
    methods = {"log_nnls": lognnls, "linear_nnls_rowscaled": w_rs, "cibersort_hpa": cib, "music_tran_nac": mus}
    # OL lineage tests
    rows = []
    for name, M in methods.items():
        lin = (M.loc["OPC"] + M.loc["Oligodendrocyte"]).to_frame("OL_lineage").T
        t = A6.arcsine_tests(lin, meta, name); t["brain_cell_class"] = "OPC+Oligodendrocyte"
        rows.append(t)
    pd.concat(rows, ignore_index=True).to_csv(rl.OUT / "revision_A6_ol_lineage_tests.tsv", sep="\t", index=False)
    # concordance across methods
    common = [s for s in lognnls.columns if all(s in M.columns for M in methods.values())]
    rows = []
    names = list(methods)
    for cls in ["MSN", "Interneuron", "Astrocyte", "Oligodendrocyte", "OPC", "Microglia", "Vascular"]:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = methods[names[i]], methods[names[j]]
                if cls not in A.index or cls not in B.index:
                    continue
                a, b = A.loc[cls, common].astype(float), B.loc[cls, common].astype(float)
                r = stats.spearmanr(a, b).statistic if a.std() > 0 and b.std() > 0 else np.nan
                rows.append(dict(brain_cell_class=cls, method_a=names[i], method_b=names[j], spearman=r,
                                 frac_nonzero_a=float((a > 1e-4).mean()), frac_nonzero_b=float((b > 1e-4).mean())))
    pd.DataFrame(rows).to_csv(rl.OUT / "revision_A6_method_concordance.tsv", sep="\t", index=False)
    comp = pd.DataFrame({n: M.mean(axis=1) for n, M in methods.items()}).fillna(0)
    comp.index.name = "brain_cell_class"
    comp.to_csv(rl.OUT / "revision_A6_method_mean_composition.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(comp.round(3))
    print(pd.read_csv(rl.OUT / "revision_A6_ol_lineage_tests.tsv", sep="\t")[["method", "brain_region", "beta", "ci_low", "ci_high", "p", "fdr_region"]].round(4).to_string(index=False))
    print(t_rs[t_rs.brain_cell_class.isin(["OPC", "Oligodendrocyte"])][["brain_region", "brain_cell_class", "beta", "ci_low", "ci_high", "p", "fdr_region"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
