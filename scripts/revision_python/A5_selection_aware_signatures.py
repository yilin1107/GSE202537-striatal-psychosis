"""A5: selection-aware inference for the five directional signatures.

The manuscript assigned each signature gene's direction from the sign of its pooled-psychosis coefficient in the
region with the smallest P value (scripts/07 + 09), then re-tested the resulting scores in the same data.
Here the whole 'direction assignment -> scoring -> regional test' chain is re-run under donor-level permutation of
the psychosis label to obtain selection-aware P values, and directions are transferred across regions.

Outputs (out/):
  revision_A5_signature_selection_aware.tsv   observed t, naive permutation P, selection-aware P per signature x region
  revision_A5_signature_cross_region.tsv      directions defined in one region (or in the other two), tested in another
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import revision_lib as rl

SIGNATURES = {
    "Activin-SMAD remodeling": ["INHBA", "SMAD1", "SMAD3", "ACVR1C", "SMURF1", "GDF5", "GDF11"],
    "Dopamine-adenosine dysregulation": ["TH", "MAOB", "ADK", "ADORA1", "ADORA2A", "IL6"],
    "Glutamate-calcium stress": ["GRM8", "GNAO1", "XBP1", "SLC2A1", "CACNA2D1", "HSPA5", "CACNA1C"],
    "Immune-NF-kB cytokine tone": ["NFKBIA", "CEBPB", "CXCL8", "PRKCZ", "TIMP2", "CCL2", "IL1B"],
    "APP processing shift": ["PSEN1", "KAT5", "ADAM17", "APP", "BACE1", "ADAM10", "PSEN2"],
}
GENES = [g for gs in SIGNATURES.values() for g in gs]


def region_fit_matrix(Y: np.ndarray, X: np.ndarray, j: int):
    """OLS for many responses at once; returns beta_j, t_j, p_j."""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y.T  # p x G
    R = Y.T - X @ B
    s2 = np.sum(R ** 2, axis=0) / (n - p)
    se = np.sqrt(XtX_inv[j, j] * s2)
    t = B[j] / se
    return B[j], t, 2 * stats.t.sf(np.abs(t), n - p)


def main(B: int = 1000, seed: int = 20260919):
    rng = np.random.default_rng(seed)
    meta = rl.load_metadata()
    logcpm = pd.read_csv(rl.PROCESSED / "gse202537_core_gene_logcpm.csv", index_col=0)
    missing = [g for g in GENES if g not in logcpm.index]
    assert not missing, missing
    z = logcpm.loc[GENES]
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0)  # z across all 215 samples (as in scripts/09)
    # region designs (pooled) for the observed labels
    designs = {}
    for region in rl.REGIONS:
        m = meta[meta.brain_region == region].reset_index(drop=True)
        X, _ = rl.region_design(m, diagnosis="pooled")
        designs[region] = (m, X)
    gene_idx = {g: i for i, g in enumerate(GENES)}
    sig_slices = {s: np.array([gene_idx[g] for g in gs]) for s, gs in SIGNATURES.items()}

    def assign_directions(labels_by_region: dict[str, np.ndarray]) -> np.ndarray:
        """direction per gene from the region with the smallest pooled-model P (as in scripts/07 & 09)."""
        betas = np.zeros((len(GENES), 3)); ps = np.ones((len(GENES), 3))
        for k, region in enumerate(rl.REGIONS):
            m, X = designs[region]
            Xp = X.to_numpy(dtype=float).copy()
            Xp[:, 1] = labels_by_region[region]
            Y = logcpm.loc[GENES, X.index].to_numpy(dtype=float)
            b, t, p = region_fit_matrix(Y, Xp, 1)
            betas[:, k] = b; ps[:, k] = p
        best = np.argmin(ps, axis=1)
        d = np.sign(betas[np.arange(len(GENES)), best])
        d[d == 0] = 1
        return d

    def directions_from_regions(labels_by_region, regions: list[str]) -> np.ndarray:
        betas = np.zeros((len(GENES), len(regions))); ps = np.ones((len(GENES), len(regions)))
        for k, region in enumerate(regions):
            m, X = designs[region]
            Xp = X.to_numpy(dtype=float).copy(); Xp[:, 1] = labels_by_region[region]
            Y = logcpm.loc[GENES, X.index].to_numpy(dtype=float)
            b, t, p = region_fit_matrix(Y, Xp, 1)
            betas[:, k] = b; ps[:, k] = p
        best = np.argmin(ps, axis=1)
        d = np.sign(betas[np.arange(len(GENES)), best]); d[d == 0] = 1
        return d

    def score_and_test(d: np.ndarray, labels_by_region, test_regions=rl.REGIONS) -> dict[tuple[str, str], tuple[float, float, float]]:
        zs = z.to_numpy(dtype=float) * d[:, None]
        scores = np.vstack([zs[idx].mean(axis=0) for idx in sig_slices.values()])  # 5 x 215
        scores = pd.DataFrame(scores, index=list(SIGNATURES), columns=z.columns)
        out = {}
        for region in test_regions:
            m, X = designs[region]
            Xp = X.to_numpy(dtype=float).copy(); Xp[:, 1] = labels_by_region[region]
            Y = scores[X.index].to_numpy(dtype=float)
            b, t, p = region_fit_matrix(Y, Xp, 1)
            for s, bb, tt, pp in zip(SIGNATURES, b, t, p):
                out[(s, region)] = (bb, tt, pp)
        return out

    obs_labels = {r: designs[r][1]["is_psychosis"].to_numpy(dtype=float) for r in rl.REGIONS}
    d_obs = assign_directions(obs_labels)
    # check against the manuscript definitions
    defs = pd.read_csv(rl.TABLES / "figure4_signature_definitions.tsv", sep="\t").set_index("gene")["direction"]
    assert all(int(d_obs[gene_idx[g]]) == int(defs[g]) for g in GENES), "direction mismatch vs manuscript"
    obs = score_and_test(d_obs, obs_labels)

    # donor-level permutation of the psychosis label
    donors = meta.drop_duplicates("patients_id")[["patients_id", "group"]].reset_index(drop=True)
    donor_psy = (donors.group != "Control").astype(float).to_numpy()
    t_sel = np.zeros((B, 5, 3)); t_naive = np.zeros((B, 5, 3))
    for b in range(B):
        perm = rng.permutation(donor_psy)
        lab_map = dict(zip(donors.patients_id, perm))
        labels = {}
        for region in rl.REGIONS:
            m, X = designs[region]
            labels[region] = m.set_index("sample_key").loc[X.index, "patients_id"].map(lab_map).to_numpy(dtype=float)
        d_perm = assign_directions(labels)
        res_sel = score_and_test(d_perm, labels)
        res_naive = score_and_test(d_obs, labels)
        for i, s in enumerate(SIGNATURES):
            for k, region in enumerate(rl.REGIONS):
                t_sel[b, i, k] = res_sel[(s, region)][1]
                t_naive[b, i, k] = res_naive[(s, region)][1]
    rows = []
    for i, s in enumerate(SIGNATURES):
        for k, region in enumerate(rl.REGIONS):
            beta, t, p = obs[(s, region)]
            rows.append(dict(signature=s, brain_region=region, beta_psychosis=beta, t_value=t, parametric_p=p,
                             naive_perm_p=(1 + np.sum(np.abs(t_naive[:, i, k]) >= abs(t))) / (B + 1),
                             selection_aware_p=(1 + np.sum(np.abs(t_sel[:, i, k]) >= abs(t))) / (B + 1),
                             selection_aware_p_onesided_pos=(1 + np.sum(t_sel[:, i, k] >= t)) / (B + 1),
                             null_mean_t_selection=float(t_sel[:, i, k].mean()), null_sd_t_selection=float(t_sel[:, i, k].std(ddof=1)),
                             null_mean_t_naive=float(t_naive[:, i, k].mean()), null_sd_t_naive=float(t_naive[:, i, k].std(ddof=1)),
                             n_perm=B))
    res = pd.DataFrame(rows)
    res["selection_aware_fdr_within_region"] = res.groupby("brain_region")["selection_aware_p"].transform(lambda x: rl.bh(x.values))
    res.to_csv(rl.OUT / "revision_A5_signature_selection_aware.tsv", sep="\t", index=False)

    # cross-region direction transfer
    rows = []
    for src in rl.REGIONS:
        d_src = directions_from_regions(obs_labels, [src])
        agree = float(np.mean(d_src == d_obs))
        out = score_and_test(d_src, obs_labels)
        for region in rl.REGIONS:
            for s in SIGNATURES:
                beta, t, p = out[(s, region)]
                rows.append(dict(direction_source=src, test_region=region, held_out=(src != region), signature=s,
                                 beta_psychosis=beta, t_value=t, p_value=p, direction_agreement_with_manuscript=agree))
    for held in rl.REGIONS:
        others = [r for r in rl.REGIONS if r != held]
        d_o = directions_from_regions(obs_labels, others)
        agree = float(np.mean(d_o == d_obs))
        out = score_and_test(d_o, obs_labels, test_regions=[held])
        for s in SIGNATURES:
            beta, t, p = out[(s, held)]
            rows.append(dict(direction_source="+".join(others), test_region=held, held_out=True, signature=s,
                             beta_psychosis=beta, t_value=t, p_value=p, direction_agreement_with_manuscript=agree))
    cross = pd.DataFrame(rows)
    cross["fdr_within_source_region"] = cross.groupby(["direction_source", "test_region"])["p_value"].transform(lambda x: rl.bh(x.values))
    cross.to_csv(rl.OUT / "revision_A5_signature_cross_region.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string())
    print(cross[cross.held_out].round(4).to_string())


if __name__ == "__main__":
    main()
