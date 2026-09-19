"""Compare the R/Bioconductor reference outputs (scripts/65, 66) with the Python implementation outputs
(results/tables/revision_A*.tsv).  Run after scripts/65_revision_limma_voom_camera.R.

python scripts/67_revision_compare_python_vs_R.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"


def compare_genes():
    py = pd.read_csv(T / "revision_A1_voom_gene_results.tsv.gz", sep="\t")
    r = pd.read_csv(T / "revision_R_A1_voom_gene_results.tsv", sep="\t")
    m = py.merge(r, on=["gene", "brain_region", "contrast"], suffixes=("_py", "_R"))
    rows = []
    for (region, contrast), d in m.groupby(["brain_region", "contrast"]):
        rows.append(dict(brain_region=region, contrast=contrast, n=len(d),
                         max_abs_diff_logFC=float((d.logFC_py - d.logFC_R).abs().max()),
                         max_abs_diff_t=float((d.t_py - d.t_R).abs().max()),
                         pearson_t=stats.pearsonr(d.t_py, d.t_R).statistic,
                         fdr10_py=int((d["adj.P.Val_py"] < 0.1).sum()), fdr10_R=int((d["adj.P.Val_R"] < 0.1).sum()),
                         fdr10_overlap=int(((d["adj.P.Val_py"] < 0.1) & (d["adj.P.Val_R"] < 0.1)).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(T / "revision_compare_A1_python_vs_R.tsv", sep="\t", index=False)
    print(out.to_string())


def compare_hallmark():
    py = pd.read_csv(T / "revision_A3_hallmark_camera_fry.tsv", sep="\t")
    r = pd.read_csv(T / "revision_R_A3_hallmark_camera_fry.tsv", sep="\t")
    m = py.merge(r, on=["brain_region", "contrast", "set"], suffixes=("_py", "_R"))
    rows = []
    for (region, contrast), d in m.groupby(["brain_region", "contrast"]):
        rows.append(dict(brain_region=region, contrast=contrast, n_sets=len(d),
                         spearman_camera_p=stats.spearmanr(d.camera_p_py, d.camera_p_R).statistic,
                         max_abs_diff_log10_camera_p=float((np.log10(d.camera_p_py) - np.log10(d.camera_p_R)).abs().max()),
                         camera_fdr10_py=int((d.camera_fdr_py < 0.1).sum()), camera_fdr10_R=int((d.camera_fdr_R < 0.1).sum()),
                         camera_fdr10_overlap=int(((d.camera_fdr_py < 0.1) & (d.camera_fdr_R < 0.1)).sum()),
                         fry_fdr10_py=int((d.fry_fdr_py < 0.1).sum()), fry_fdr10_R=int((d.fry_fdr_R < 0.1).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(T / "revision_compare_A3_python_vs_R.tsv", sep="\t", index=False)
    print(out.to_string())


def compare_sensitivity():
    py = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(T.glob("revision_A2_run_sensitivity_gene_results_*.tsv.gz"))], ignore_index=True)
    r = pd.read_csv(T / "revision_R_A2_run_sensitivity_gene_results.tsv", sep="\t")
    m = py.merge(r, on=["variant", "gene", "brain_region", "contrast"], suffixes=("_py", "_R"))
    rows = []
    for (variant, region, contrast), d in m.groupby(["variant", "brain_region", "contrast"]):
        rows.append(dict(variant=variant, brain_region=region, contrast=contrast, n=len(d),
                         pearson_t=stats.pearsonr(d.t_py, d.t_R).statistic,
                         fdr10_py=int((d["adj.P.Val_py"] < 0.1).sum()), fdr10_R=int((d["adj.P.Val_R"] < 0.1).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(T / "revision_compare_A2_python_vs_R.tsv", sep="\t", index=False)
    print(out.to_string())


if __name__ == "__main__":
    for f in (compare_genes, compare_hallmark, compare_sensitivity):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped:", e)
