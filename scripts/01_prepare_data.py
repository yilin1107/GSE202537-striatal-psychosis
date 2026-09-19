from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


GSE_ID = "GSE202537"
SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/"
    "GSE202537/matrix/GSE202537_series_matrix.txt.gz"
)
READ_COUNTS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/"
    "GSE202537/suppl/GSE202537_Read_counts_NAc_Caudate_Putamen_psychosis_mcontrols.csv.gz"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
RAW_MATRIX = RAW_DIR / "GSE202537_series_matrix.txt.gz"
RAW_COUNTS = RAW_DIR / "GSE202537_Read_counts_NAc_Caudate_Putamen_psychosis_mcontrols.csv.gz"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def parse_series_matrix(path: Path) -> pd.DataFrame:
    sample_rows: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_"):
                sample_rows.append(next(csv.reader([line.rstrip("\n")], delimiter="\t")))

    accession_row = next(row for row in sample_rows if row[0] == "!Sample_geo_accession")
    accessions = [value.strip('"') for value in accession_row[1:]]
    records = [{"geo_accession": accession} for accession in accessions]

    for row in sample_rows:
        key = row[0].replace("!Sample_", "")
        values = [value.strip('"') for value in row[1:]]
        if len(values) != len(records):
            continue
        if key == "characteristics_ch1":
            for record, value in zip(records, values):
                if ":" not in value:
                    continue
                field, field_value = value.split(":", 1)
                normalized = (
                    field.strip()
                    .lower()
                    .replace(" ", "_")
                    .replace(".", "_")
                    .replace("/", "_")
                )
                record[normalized] = field_value.strip()
        elif key in {"title", "source_name_ch1"}:
            for record, value in zip(records, values):
                record[key] = value

    samples = pd.DataFrame.from_records(records)
    samples["sample_key"] = samples["title"].map(
        lambda value: re.search(r"\[(.*?)\]", str(value)).group(1)
        if re.search(r"\[(.*?)\]", str(value))
        else ""
    )
    samples["brain_region"] = samples["tissue"].replace({"Nac": "NAc"})
    samples["group"] = samples["disease_state"].map(
        {
            "match control": "Control",
            "psychosis_schizophrenia": "SCZ",
            "psychosis_bipolar": "BD with psychosis",
        }
    )

    numeric_columns = [
        "corrected_tod",
        "age",
        "pmi",
        "library_size",
        "rin",
        "bmi",
        "ph",
        "tissuestoragetime",
    ]
    for column in numeric_columns:
        if column in samples.columns:
            samples[column] = pd.to_numeric(samples[column].replace({"NA": np.nan}), errors="coerce")

    samples["corrected_tod_24h"] = samples["corrected_tod"] % 24
    return samples


def pca_from_logcpm(logcpm: pd.DataFrame, n_components: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = logcpm.T.to_numpy(dtype=float)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u_matrix, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    scores = u_matrix[:, :n_components] * singular_values[:n_components]
    variance = singular_values**2
    explained = variance / variance.sum()

    pca = pd.DataFrame(
        scores,
        index=logcpm.columns,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    )
    variance_df = pd.DataFrame(
        {
            "pc": [f"PC{i}" for i in range(1, n_components + 1)],
            "variance_explained": explained[:n_components],
        }
    )
    return pca, variance_df


def association_p_value(values: pd.Series, covariate: pd.Series, kind: str) -> tuple[float, float]:
    aligned = pd.concat([values, covariate], axis=1).dropna()
    if aligned.empty:
        return np.nan, np.nan
    y_values = aligned.iloc[:, 0]
    x_values = aligned.iloc[:, 1]
    if kind == "continuous":
        if x_values.nunique() < 3:
            return np.nan, np.nan
        statistic, p_value = stats.spearmanr(x_values, y_values)
        return float(statistic), float(p_value)

    groups = [y_values.loc[x_values == level] for level in sorted(x_values.dropna().unique())]
    groups = [group for group in groups if len(group) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    statistic, p_value = stats.kruskal(*groups, nan_policy="omit")
    return float(statistic), float(p_value)


def prepare_expression_qc(samples: pd.DataFrame) -> list[Path]:
    counts = pd.read_csv(RAW_COUNTS, index_col=0)
    sample_lookup = samples.set_index("sample_key", drop=False)
    missing = sorted(set(counts.columns) - set(sample_lookup.index))
    if missing:
        raise ValueError(f"Count matrix samples missing from metadata: {missing[:5]}")

    ordered_samples = sample_lookup.loc[counts.columns].reset_index(drop=True)
    count_depth = counts.sum(axis=0)
    detected_genes = (counts > 0).sum(axis=0)
    keep_genes = (counts >= 10).sum(axis=1) >= max(3, int(np.ceil(0.10 * counts.shape[1])))
    filtered_counts = counts.loc[keep_genes]
    cpm = filtered_counts.div(count_depth, axis=1) * 1_000_000
    logcpm = np.log2(cpm + 1)
    pca, variance_df = pca_from_logcpm(logcpm)

    qc = ordered_samples.copy()
    qc["count_depth"] = qc["sample_key"].map(count_depth)
    qc["count_depth_million"] = qc["count_depth"] / 1_000_000
    qc["detected_genes"] = qc["sample_key"].map(detected_genes)
    qc["genes_retained_for_pca"] = int(keep_genes.sum())

    pca = pca.reset_index(names="sample_key").merge(
        qc[
            [
                "sample_key",
                "geo_accession",
                "patients_id",
                "brain_region",
                "group",
                "gender",
                "age",
                "pmi",
                "rin",
                "ph",
                "count_depth_million",
                "corrected_tod_24h",
                "sequence_id",
            ]
        ],
        on="sample_key",
        how="left",
    )

    covariates = [
        ("brain_region", "Brain region", "categorical"),
        ("group", "Diagnostic group", "categorical"),
        ("gender", "Sex", "categorical"),
        ("sequence_id", "Sequence run", "categorical"),
        ("age", "Age", "continuous"),
        ("pmi", "PMI", "continuous"),
        ("rin", "RIN", "continuous"),
        ("ph", "Brain pH", "continuous"),
        ("count_depth_million", "Count depth", "continuous"),
        ("corrected_tod_24h", "Corrected TOD", "continuous"),
    ]
    association_rows: list[dict[str, str | float]] = []
    pca_indexed = pca.set_index("sample_key")
    for pc in [f"PC{i}" for i in range(1, 6)]:
        for column, label, kind in covariates:
            statistic, p_value = association_p_value(pca_indexed[pc], pca_indexed[column], kind)
            association_rows.append(
                {
                    "pc": pc,
                    "covariate": label,
                    "test": "Spearman" if kind == "continuous" else "Kruskal-Wallis",
                    "statistic": statistic,
                    "p_value": p_value,
                    "minus_log10_p": -np.log10(p_value) if p_value and p_value > 0 else np.nan,
                }
            )
    associations = pd.DataFrame(association_rows)

    output_paths = [
        PROCESSED_DIR / "gse202537_expression_qc.csv",
        PROCESSED_DIR / "gse202537_pca_coordinates.csv",
        TABLE_DIR / "supplementary_figure1_pca_variance.tsv",
        TABLE_DIR / "supplementary_figure1_pc_covariate_associations.tsv",
    ]
    qc.to_csv(output_paths[0], index=False)
    pca.to_csv(output_paths[1], index=False)
    variance_df.to_csv(output_paths[2], sep="\t", index=False)
    associations.to_csv(output_paths[3], sep="\t", index=False)
    return output_paths


def make_subject_metadata(samples: pd.DataFrame) -> pd.DataFrame:
    first_columns = {
        "group": "first",
        "disease_state": "first",
        "age": "first",
        "gender": "first",
        "race": "first",
        "pmi": "first",
        "bmi": "first",
        "ph": "first",
        "tissuestoragetime": "first",
        "manner_of_death": "first",
        "corrected_tod": "first",
        "corrected_tod_24h": "first",
        "rin": "mean",
    }
    subjects = (
        samples.sort_values(["patients_id", "brain_region"])
        .groupby("patients_id", as_index=False)
        .agg(first_columns)
        .rename(columns={"patients_id": "subject_id", "gender": "sex"})
        .reset_index(drop=True)
    )
    return subjects


def format_median_iqr(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return ""
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    return f"{clean.median():.1f} ({q1:.1f}-{q3:.1f})"


def clinical_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    group_order = ["Control", "SCZ", "BD with psychosis"]

    for group in group_order:
        subset = subjects.loc[subjects["group"] == group]
        rows.append(
            {
                "group": group,
                "donors": str(len(subset)),
                "age_median_iqr": format_median_iqr(subset["age"]),
                "pmi_median_iqr": format_median_iqr(subset["pmi"]),
                "rin_median_iqr": format_median_iqr(subset["rin"]),
                "brain_ph_median_iqr": format_median_iqr(subset["ph"]),
                "bmi_median_iqr": format_median_iqr(subset["bmi"]),
                "female_n": str((subset["sex"] == "Female").sum()),
                "male_n": str((subset["sex"] == "Male").sum()),
                "white_n": str((subset["race"] == "White").sum()),
                "black_n": str((subset["race"] == "Black").sum()),
            }
        )
    return pd.DataFrame(rows)


def p_value_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def clinical_tests(subjects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    group_order = ["Control", "SCZ", "BD with psychosis"]

    continuous = {
        "age": "Age",
        "pmi": "Postmortem interval",
        "rin": "RNA integrity number",
        "ph": "Brain pH",
        "bmi": "Body mass index",
        "tissuestoragetime": "Tissue storage time",
        "corrected_tod_24h": "Corrected time of death",
    }
    for column, label in continuous.items():
        grouped = [
            subjects.loc[subjects["group"] == group, column].dropna()
            for group in group_order
        ]
        if sum(len(values) > 0 for values in grouped) >= 2:
            statistic, p_value = stats.kruskal(*grouped, nan_policy="omit")
        else:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "variable": label,
                "test": "Kruskal-Wallis",
                "statistic": f"{statistic:.4g}" if not pd.isna(statistic) else "NA",
                "p_value": p_value_text(float(p_value)) if not pd.isna(p_value) else "NA",
            }
        )

    for column, label in {"sex": "Sex", "race": "Race", "manner_of_death": "Manner of death"}.items():
        table = pd.crosstab(subjects[column], subjects["group"])
        statistic, p_value, dof, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "variable": label,
                "test": "Chi-square",
                "statistic": f"{statistic:.4g}",
                "p_value": p_value_text(float(p_value)),
            }
        )
    return pd.DataFrame(rows)


def write_checksums(paths: list[Path], output_path: Path) -> None:
    lines = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSE202537 clinical metadata.")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.force_download and RAW_MATRIX.exists():
        RAW_MATRIX.unlink()
    if args.force_download and RAW_COUNTS.exists():
        RAW_COUNTS.unlink()

    download(SERIES_MATRIX_URL, RAW_MATRIX)
    download(READ_COUNTS_URL, RAW_COUNTS)
    samples = parse_series_matrix(RAW_MATRIX)
    subjects = make_subject_metadata(samples)
    summary = clinical_summary(subjects)
    tests = clinical_tests(subjects)
    expression_outputs = prepare_expression_qc(samples)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    samples.to_csv(PROCESSED_DIR / "gse202537_sample_metadata.csv", index=False)
    subjects.to_csv(PROCESSED_DIR / "gse202537_subject_metadata.csv", index=False)
    summary.to_csv(TABLE_DIR / "figure1_clinical_summary.tsv", sep="\t", index=False)
    tests.to_csv(TABLE_DIR / "figure1_clinical_tests.tsv", sep="\t", index=False)
    write_checksums(
        [
            RAW_MATRIX,
            RAW_COUNTS,
            PROCESSED_DIR / "gse202537_sample_metadata.csv",
            PROCESSED_DIR / "gse202537_subject_metadata.csv",
            TABLE_DIR / "figure1_clinical_summary.tsv",
            TABLE_DIR / "figure1_clinical_tests.tsv",
            *expression_outputs,
        ],
        ROOT / "checksums.txt",
    )

    print(f"Prepared {len(samples)} samples and {len(subjects)} donors.")


if __name__ == "__main__":
    main()
