# Changelog

## 3.0.1 documentation update - September 2026

- Clarified script headers, README text, and the Supplementary Data S1 worksheet index; figure references in the index now follow the article numbering.
- Simplified `scripts/03_validate_outputs.py` to figure-file checks and removed a manuscript-checking utility (`scripts/55`); neither contributes to the analyses.
- Numerical results are unchanged.

## 3.0.1 - 2026-09-20

- Unified six downstream statistical summaries with the unchanged primary R limma-voom model used in Figure 1. The primary model and all 124,860 region-contrast gene-model rows were verified against a fresh R fit; Supplementary Data S2 was not changed.
- Corrected SCZ-selected gene counts in current Supplementary Figure S3 to 141 in caudate and 2152 in putamen (previous downstream summary: 146 and 2142).
- Recomputed diagnosis-specific concordance, descriptive BD power, regional-gradient diagnostics, cross-region concordance, OLS comparison summaries, and model information from that same source. Power remains a plug-in descriptive estimate, not evidence for a transdiagnostic effect.
- Updated current Supplementary Figures S3 and S5 and six worksheets in Supplementary Data S1; all main figures were retained.
- Added model-verification, summary-reconciliation, and portable S3/S5 plotting workflows, with assertions that reject mixed primary-model versions.
- Added a supplementary-figure numbering crosswalk. Current S3/S5 correspond to S12/S9 in release 3.0.0.
- Scientific scope is unchanged: bulk observational associations, reference-dependent deconvolution, fixed-membership signature permutation, and no statistical colocalization or independent cross-region replication.

## 3.0.0 - 2026-09-19

Initial public release of the revision analysis workflows.
