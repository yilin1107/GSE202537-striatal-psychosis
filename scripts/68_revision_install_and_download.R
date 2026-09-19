#!/usr/bin/env Rscript
# 68_revision_install_and_download.R
# Step 0 of the revision R run: make sure the Bioconductor packages are present and fetch the
# Tran et al. 2021 NAc single-nucleus reference (SingleCellExperiment, 8 donors) used by script 66.
options(timeout = 7200, warn = 1)
cran <- "https://cloud.r-project.org"
cat("R version:", R.version.string, "\n")
cat("library paths:", paste(.libPaths(), collapse = " ; "), "\n")

have <- function(p) requireNamespace(p, quietly = TRUE)
bioc_pkgs <- c("edgeR", "limma", "RUVSeq", "sva", "SingleCellExperiment")
missing <- bioc_pkgs[!vapply(bioc_pkgs, have, logical(1))]
if (length(missing) > 0) {
  cat("Installing missing Bioconductor packages:", paste(missing, collapse = ", "), "\n")
  if (!have("BiocManager")) install.packages("BiocManager", repos = cran)
  tryCatch(BiocManager::install(missing, ask = FALSE, update = FALSE),
           error = function(e) cat("BiocManager::install failed:", conditionMessage(e), "\n"))
}
if (!have("MuSiC")) {
  cat("MuSiC not found; trying remotes::install_github('xuranw/MuSiC')\n")
  if (!have("remotes")) install.packages("remotes", repos = cran)
  tryCatch(remotes::install_github("xuranw/MuSiC", upgrade = "never"),
           error = function(e) cat("MuSiC install failed:", conditionMessage(e), "\n"))
}
for (p in c(bioc_pkgs, "MuSiC", "IOBR", "Matrix")) cat(sprintf("%-22s %s\n", p, if (have(p)) as.character(packageVersion(p)) else "MISSING"))

# Tran et al. 2021 (Neuron) NAc snRNA-seq reference, hosted by LIBD (see github.com/LieberInstitute/10xPilot_snRNAseq-human)
url <- "https://libd-snrnaseq-pilot.s3.us-east-2.amazonaws.com/SCE_NAc-n8_tran-etal.rda"
dest <- file.path("data", "raw", "SCE_NAc-n8_tran-etal.rda")
if (!file.exists(dest) || file.info(dest)$size < 1e6) {
  cat("Downloading", url, "\n")
  ok <- tryCatch({ download.file(url, dest, mode = "wb", quiet = FALSE); TRUE },
                 error = function(e) { cat("download failed:", conditionMessage(e), "\n"); FALSE })
  if (ok) cat("Downloaded", dest, "size (MB):", round(file.info(dest)$size / 1e6, 1), "\n")
} else {
  cat("Reference already present:", dest, "size (MB):", round(file.info(dest)$size / 1e6, 1), "\n")
}
cat("Step 68 finished.\n")
