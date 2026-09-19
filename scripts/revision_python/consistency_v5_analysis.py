"""Reconcile primary-model dependent summaries without changing the primary results."""
from pathlib import Path
import os
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy import stats
from A1_limma_voom import concordance
from A4_diagnosis_specific import genomewide_concordance
from B2_gradient_diagnostics import storey_pi0

ROOT = Path(os.environ.get('GSE202537_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
T = ROOT / 'results/tables'
OUT = ROOT / 'results/revision_consistency_v5'
v = pd.read_csv(T / 'revision_R_A1_voom_gene_results.tsv', sep='\t')
u = pd.read_csv(OUT / 'primary_gene_uncertainty.tsv', sep='\t')
REGIONS = ['NAc', 'Caudate', 'Putamen']
outputs = {}

def emit(name, data):
    data.to_csv(OUT / name, sep='\t', index=False)
    outputs[name] = data

def region_frame(region, contrast='SCZ_vs_Control'):
    return v[(v.brain_region == region) & (v.contrast == contrast)].set_index('gene')

emit('revision_A4_scz_bd_genomewide_concordance.tsv', genomewide_concordance(v))
power_rows = []
old_power = pd.read_csv(T / 'revision_A4_bd_power.tsv', sep='\t').set_index('brain_region')
for region in REGIONS:
    s = region_frame(region)
    unc = u[u.brain_region == region].set_index('gene').loc[s.index]
    df = unc.df_total.to_numpy()
    scz_se, bd_se = unc.se_scz.to_numpy(), unc.se_bd.to_numpy()
    # The original plug-in power and approximate 80% MDE definitions are retained.
    tc = stats.t.ppf(.975, df)
    mde_bd = (tc + stats.t.ppf(.8, df)) * bd_se
    mde_scz = (tc + stats.t.ppf(.8, df)) * scz_se
    sig = (s['adj.P.Val'] < .1).to_numpy()
    p_thr = s.loc[sig, 'P.Value'].max() if sig.any() else np.nan
    old = old_power.loc[region]
    r = dict(brain_region=region, n_bd=int(old.n_bd), n_scz=int(old.n_scz), n_control=int(old.n_control),
        df_total=float(np.median(df)), median_se_scz=float(np.median(scz_se)), median_se_bd=float(np.median(bd_se)),
        se_ratio_bd_over_scz=float(np.median(bd_se/scz_se)),
        mde_logfc_80pct_alpha05_scz_median=float(np.median(mde_scz)),
        mde_logfc_80pct_alpha05_bd_median=float(np.median(mde_bd)), n_scz_fdr10=int(sig.sum()),
        scz_fdr10_p_threshold=p_thr)
    if sig.any():
        delta = s.loc[sig, 'logFC'].abs().to_numpy()
        def power(critical):
            ncp = delta / bd_se[sig]
            return stats.nct.sf(critical, df[sig], ncp) + stats.nct.cdf(-critical, df[sig], ncp)
        pw = power(tc[sig])
        pw_fdr = power(stats.t.ppf(1-p_thr/2, df[sig]))
        r.update(median_abs_logfc_scz_fdr10=float(np.median(delta)), bd_power_alpha05_median=float(np.median(pw)),
            bd_power_alpha05_frac_ge80=float(np.mean(pw >= .8)), bd_power_fdr_equiv_median=float(np.median(pw_fdr)),
            bd_power_fdr_equiv_frac_ge80=float(np.mean(pw_fdr >= .8)),
            frac_scz_fdr10_effects_below_bd_mde=float(np.mean(delta < mde_bd[sig])))
    power_rows.append(r)
emit('revision_A4_bd_power.tsv', pd.DataFrame(power_rows))

diag = pd.read_csv(T / 'revision_B2_region_gradient_diagnostics.tsv', sep='\t').set_index('brain_region')
cross = []
for region in REGIONS:
    s = region_frame(region)
    ps, ph = storey_pi0(s['P.Value'].to_numpy())
    hist, _ = np.histogram(s['P.Value'], bins=np.linspace(0, 1, 11))
    changes = dict(voom_scz_fdr10=int((s['adj.P.Val'] < .1).sum()),
        voom_scz_nominal_p05=int((s['P.Value'] < .05).sum()), pi0_storey_smoother_voom=ps,
        pi0_lambda05_voom=ph, est_n_nonnull_voom=int(round((1-ps)*len(s))),
        p_hist_bin1_frac=hist[0]/len(s), p_hist_bin10_frac=hist[-1]/len(s),
        median_residual_sd_voom=float(u.loc[u.brain_region == region, 'residual_sd'].median()),
        median_abs_t_scz=float(s.t.abs().median()), mean_abs_logfc_scz=float(s.logFC.abs().mean()))
    for key, value in changes.items():
        diag.loc[region, key] = value
    sig = s.index[s['adj.P.Val'] < .1]
    for target in REGIONS:
        if target == region:
            continue
        t = region_frame(target).loc[s.index]
        diag.loc[region, f'spearman_t_with_{target}'] = stats.spearmanr(s.t, t.t).statistic
        if len(sig):
            ss, tt = s.loc[sig], t.loc[sig]
            agree = np.sign(ss.logFC) == np.sign(tt.logFC)
            cross.append(dict(source_region=region, target_region=target, n_source_fdr10=len(sig),
                sign_concordance=float(agree.mean()), binom_p=stats.binomtest(int(agree.sum()), len(sig), .5).pvalue,
                frac_nominal_p05_in_target=float((tt['P.Value'] < .05).mean()),
                frac_fdr10_in_target=float((tt['adj.P.Val'] < .1).mean()),
                spearman_logfc=stats.spearmanr(ss.logFC, tt.logFC).statistic,
                median_abs_logfc_source=float(ss.logFC.abs().median()), median_abs_logfc_target=float(tt.logFC.abs().median())))
emit('revision_B2_region_gradient_diagnostics.tsv', diag.reset_index())
emit('revision_B2_cross_region_concordance.tsv', pd.DataFrame(cross))
ols = pd.read_csv(T / 'npj_figure1_gene_ma_results.tsv', sep='\t')
emit('revision_A1_voom_vs_ols_summary.tsv', concordance(v, ols))
info = pd.read_csv(OUT / 'primary_model_info.tsv', sep='\t').rename(columns={'df_residual':'df_res', 's2_prior':'var_prior'})
emit('revision_A1_model_info.tsv', info)

for name in ['revision_A5_signature_selection_aware.tsv', 'revision_A5_signature_cross_region.tsv',
             'revision_R_A2_run_sensitivity_summary.tsv']:
    shutil.copy2(T/name, OUT/name)

# Track every affected cell so the scope of the correction can be independently checked.
changes = []
for name, new in outputs.items():
    old = pd.read_csv(T / name, sep='\t')
    assert list(new.columns) == list(old.columns), name
    assert new.shape == old.shape, name
    for i in range(len(new)):
        for col in new.columns:
            a, b = old.iloc[i][col], new.iloc[i][col]
            equal = (pd.isna(a) and pd.isna(b)) or a == b
            if not equal:
                changes.append({'table': name, 'row': i+2, 'column': col, 'old': a, 'new': b})
pd.DataFrame(changes).to_csv(OUT/'changed_summary_cells.csv', index=False)
sources = [T/'revision_A1_voom_gene_results.tsv.gz', T/'revision_R_A1_voom_gene_results.tsv']
manifest = {'primary_source': sources[1].name, 'old_source': sources[0].name,
    'source_files': [{ 'file': p.name, 'modified': pd.Timestamp(p.stat().st_mtime, unit='s', tz='UTC').tz_convert('Asia/Taipei').isoformat(),
                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
    'updated_tables': list(outputs), 'changed_summary_cells': len(changes)}
(OUT/'consistency_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
for name in ['revision_A4_scz_bd_genomewide_concordance.tsv', 'revision_A4_bd_power.tsv',
             'revision_B2_region_gradient_diagnostics.tsv', 'revision_B2_cross_region_concordance.tsv']:
    print(name, '\n', outputs[name].round(6).to_string(index=False))
