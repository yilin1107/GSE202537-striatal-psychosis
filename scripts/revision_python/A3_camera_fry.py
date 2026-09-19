"""A3: correlation-aware Hallmark gene-set testing (camera, fry) on the voom fits from A1.

Outputs (out/):
  revision_A3_hallmark_camera_fry.tsv   per set x region x contrast: original permutation z/P/FDR,
                                        camera (fixed 0.01), camera (estimated cor), fry directional
  revision_A3_hallmark_method_summary.tsv  counts of FDR<0.10 sets per region x contrast x method
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import revision_lib as rl


def main():
    sc = rl.load_symbol_counts()
    genes = sc.index
    defs = pd.read_csv(rl.TABLES / "npj_figure1_hallmark_definitions.tsv", sep="\t")
    orig = pd.read_csv(rl.TABLES / "npj_figure1_hallmark_enrichment.tsv", sep="\t")
    gene_pos = {g: i for i, g in enumerate(genes)}
    # gene membership from the GMT (same parsing as scripts/32: upper-case, de-duplicated)
    gmt = {}
    with open(rl.RAW / "MSigDB_Hallmark_2020.gmt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gmt[parts[0].strip()] = sorted({g.strip().upper() for g in parts[2:] if g.strip()})
    sets = {}
    for _, row in defs.iterrows():
        idx = np.array([gene_pos[g] for g in gmt[row["hallmark"]] if g in gene_pos])
        if len(idx) >= 5:
            sets[row["hallmark_id"]] = idx
    # sanity: overlap sizes must match the original table
    chk = orig[(orig.brain_region == "Putamen") & (orig.contrast == "SCZ_vs_Control")].set_index("hallmark_id")["overlap_genes"]
    assert all(len(sets[k]) == chk[k] for k in sets), "Hallmark overlap mismatch vs original table"
    print("Hallmark sets:", len(sets))
    rows = []
    for region in rl.REGIONS:
        E = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_E.npy")
        W = np.load(rl.INTERMEDIATE / f"A1_fit_{region}_W.npy")
        X = pd.read_csv(rl.INTERMEDIATE / f"A1_fit_{region}_design.tsv", sep="\t", index_col=0)
        cols = list(X.columns)
        for col, contrast in [("is_scz", "SCZ_vs_Control"), ("is_bd_psychosis", "BD_psychosis_vs_Control")]:
            j = cols.index(col)
            unscaledt, U, df_res = rl.lm_effects(E, X.to_numpy(), j, weights=W)
            cam_fixed, modt, Stat = rl.camera(unscaledt, U, df_res, sets, inter_gene_cor=0.01)
            cam_est, _, _ = rl.camera(unscaledt, U, df_res, sets, inter_gene_cor=None)
            fr = rl.fry(unscaledt, U, df_res, sets)
            merged = cam_fixed.rename(columns={"Correlation": "camera_cor_fixed", "Direction": "camera_direction",
                                               "two_sample_t": "camera_t", "PValue": "camera_p", "FDR": "camera_fdr"})
            merged = merged.merge(cam_est[["set", "Correlation", "PValue", "FDR"]].rename(
                columns={"Correlation": "estimated_intergene_cor", "PValue": "camera_estcor_p", "FDR": "camera_estcor_fdr"}), on="set")
            merged = merged.merge(fr[["set", "t", "PValue", "FDR"]].rename(
                columns={"t": "fry_t", "PValue": "fry_p", "FDR": "fry_fdr"}), on="set")
            merged.insert(0, "contrast", contrast)
            merged.insert(0, "brain_region", region)
            # mean moderated t within set (descriptive)
            merged["mean_moderated_t_voom"] = [modt[sets[s]].mean() for s in merged["set"]]
            rows.append(merged)
            print(region, contrast, "camera FDR<0.10:", int((merged.camera_fdr < 0.1).sum()),
                  "camera(est cor) FDR<0.10:", int((merged.camera_estcor_fdr < 0.1).sum()),
                  "fry FDR<0.10:", int((merged.fry_fdr < 0.1).sum()))
    res = pd.concat(rows, ignore_index=True)
    o = orig.rename(columns={"hallmark_id": "set"}).set_index(["brain_region", "contrast", "set"])[
        ["label", "category", "overlap_genes", "mean_t", "enrichment_z", "p_value", "fdr"]]
    o.columns = ["label", "category", "orig_overlap_genes", "orig_mean_t_ols", "orig_perm_z", "orig_perm_p", "orig_perm_fdr"]
    res = res.join(o, on=["brain_region", "contrast", "set"])
    res = res[["brain_region", "contrast", "set", "label", "category", "NGenes", "orig_perm_z", "orig_perm_p", "orig_perm_fdr",
               "mean_moderated_t_voom", "camera_direction", "camera_t", "camera_cor_fixed", "camera_p", "camera_fdr",
               "estimated_intergene_cor", "camera_estcor_p", "camera_estcor_fdr", "fry_t", "fry_p", "fry_fdr"]]
    res.to_csv(rl.OUT / "revision_A3_hallmark_camera_fry.tsv", sep="\t", index=False)
    summ = res.groupby(["brain_region", "contrast"]).agg(
        n_sets=("set", "count"),
        perm_fdr10=("orig_perm_fdr", lambda x: int((x < 0.1).sum())),
        camera_fdr10=("camera_fdr", lambda x: int((x < 0.1).sum())),
        camera_estcor_fdr10=("camera_estcor_fdr", lambda x: int((x < 0.1).sum())),
        fry_fdr10=("fry_fdr", lambda x: int((x < 0.1).sum())),
        camera_fdr05=("camera_fdr", lambda x: int((x < 0.05).sum())),
        median_estimated_cor=("estimated_intergene_cor", "median"),
    ).reset_index()
    # concordance of permutation z with camera t
    conc = res.groupby(["brain_region", "contrast"]).apply(
        lambda d: pd.Series({"spearman_permz_camerat": d[["orig_perm_z", "camera_t"]].corr(method="spearman").iloc[0, 1],
                             "sets_perm_fdr10_and_camera_fdr10": int(((d.orig_perm_fdr < 0.1) & (d.camera_fdr < 0.1)).sum())})).reset_index()
    summ = summ.merge(conc, on=["brain_region", "contrast"])
    summ.to_csv(rl.OUT / "revision_A3_hallmark_method_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    main()
