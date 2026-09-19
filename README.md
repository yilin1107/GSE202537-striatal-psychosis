# GSE202537 striatal psychosis transcriptomics

Reproducible analysis code supporting the manuscript **Systems-Level Transcriptomic and Cellular Architecture of the Human Striatum in Schizophrenia and Bipolar Psychosis**.

The repository contains the Python and R workflows used to analyze GSE202537 across nucleus accumbens, caudate, and putamen. It includes expression quality control, diagnosis-specific limma-voom models, correlation-aware Hallmark testing, pathway and directional-signature analyses, immune and brain-resident reference deconvolution, external marker validation, and the revision sensitivity analyses.

## Data availability

Raw and processed transcriptomic inputs are not redistributed in this repository. The primary RNA-seq data are available from GEO under accession [GSE202537](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE202537). Supplementary statistical tables and figure-supporting values are deposited in Zenodo under concept DOI [10.5281/zenodo.20741182](https://doi.org/10.5281/zenodo.20741182); version 3.0.0 contains the files used for the Biology revision.

Additional public references used by individual workflows are documented in the scripts, including the Human Protein Atlas single-nucleus brain resource, the CELLxGENE CSF reference, the LIBD Tran et al. nucleus accumbens single-nucleus reference, GTEx v8, and the GWAS Catalog.

## Repository layout

- `scripts/01_*.py` to `scripts/16_*.py`: original data preparation, pathway, gene, signature, cell-type, and external-validation workflows.
- `scripts/32_*.py` to `scripts/49_*.py` and companion R files: Hallmark, IOBR/CIBERSORT, MuSiC, and brain-resident deconvolution workflows.
- `scripts/65_*.R` to `scripts/69_*.R`: limma-voom, camera/fry, sequencing-run sensitivity, alternative deconvolution, and comparison pipeline added during revision.
- `scripts/revision_python/`: diagnosis-specific, selection-aware, gradient, and composition sensitivity analyses plus revision figure generation.
- `src/plotting_theme.py`: shared publication-figure styling.
- `environment/`: Python requirements and R session information captured for the revision analyses.

## Setup

Run commands from the repository root. Python dependencies can be installed with:

```bash
python -m pip install -r environment/requirements.txt
```

R and Bioconductor dependencies used in the revision are listed in `environment/revision_R_sessionInfo.txt` and `environment/revision_R_A6_sessionInfo.txt`. The helper script `scripts/68_revision_install_and_download.R` installs missing packages and downloads the independent Tran et al. reference.

## Reproduction outline

1. Download GSE202537 count matrices and series metadata through the public GEO resources referenced in `scripts/01_prepare_data.py`.
2. Place external inputs under `data/raw/` and run the preparation workflow:

```bash
python scripts/01_prepare_data.py
```

3. Run the relevant original or systems-level scripts in numeric order.
4. Run the revision pipeline:

```bash
Rscript scripts/69_run_revision_all.R
```

5. Generate revision figures after the result tables are available:

```bash
python scripts/revision_python/revision_figures.py
```

Set `GSE202537_PROJECT_ROOT` when the repository root cannot be inferred from the script location. Revision analysis modules also accept `REVISION_ROOT` through `scripts/revision_python/revision_lib.py`.

## Statistical scope

The repository reproduces an observational analysis of public postmortem bulk RNA-seq data. Deconvolution outputs are reference-dependent relative weights, and genetic-regulatory annotations do not constitute statistical colocalization. See the manuscript and the Zenodo tables for the complete model definitions, multiplicity correction, sensitivity analyses, and evidence-grading rules.

## Version

This release corresponds to the Biology revision and Zenodo version 3.0.0 (19 September 2026). Documentation update (September 2026): script headers were clarified, `scripts/03_validate_outputs.py` was simplified to figure-file checks, and a manuscript-checking utility (`scripts/55`) was removed; neither contributes to the analyses, and numerical results are unchanged.
