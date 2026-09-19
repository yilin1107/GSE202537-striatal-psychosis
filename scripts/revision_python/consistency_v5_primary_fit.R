#!/usr/bin/env Rscript
# Recover uncertainty from the unchanged primary R model and verify every estimate.
suppressPackageStartupMessages({library(edgeR); library(limma)})
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(if (length(args)) args[[1]] else ".", winslash = "/")
processed_dir <- file.path(root, "data", "processed")
out <- file.path(root, "results", "revision_consistency_v5")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# Evaluate the original data preparation and design function, without its output writes.
expressions <- parse(file.path(root, "scripts", "65_revision_limma_voom_camera.R"))
lhs <- vapply(expressions, function(e) {
  if (is.call(e) && identical(e[[1]], as.name("<-"))) paste(deparse(e[[2]]), collapse = "") else ""
}, character(1))
for (i in seq(which(lhs == "counts"), which(lhs == "make_design"))) eval(expressions[[i]])
dge <- calcNormFactors(DGEList(counts = counts), method = "TMM")
primary <- read.delim(file.path(root, "results", "tables", "revision_R_A1_voom_gene_results.tsv"))
uncertainty <- list(); checks <- list(); model_info <- list()
for (region in regions) {
  idx <- which(meta$brain_region == region)
  m <- meta[idx, ]
  design <- make_design(m)
  fit <- eBayes(lmFit(voom(dge[, idx], design), design))
  stopifnot(length(unique(fit$df.total)) == 1)
  for (cc in c("is_scz", "is_bd_psychosis")) {
    contrast <- if (cc == "is_scz") "SCZ_vs_Control" else "BD_psychosis_vs_Control"
    tt <- topTable(fit, coef = cc, number = Inf, sort.by = "none")
    # Explicit indexing avoids ambiguous column/environment name lookup.
    old <- primary[primary$brain_region == region & primary$contrast == contrast, ]
    old <- old[match(rownames(tt), old$gene), ]
    stopifnot(identical(rownames(tt), old$gene))
    fields <- c("AveExpr", "logFC", "t", "P.Value", "adj.P.Val")
    diffs <- vapply(fields, function(f) max(abs(tt[[f]] - old[[f]])), numeric(1))
    stopifnot(all(diffs < 1e-10))
    checks[[length(checks) + 1]] <- data.frame(brain_region = region, contrast = contrast,
      n_genes = nrow(tt), fdr10 = sum(tt$adj.P.Val < 0.1), t(diffs), check.names = FALSE)
  }
  uncertainty[[region]] <- data.frame(gene = rownames(fit$coefficients), brain_region = region,
    se_scz = fit$stdev.unscaled[, "is_scz"] * sqrt(fit$s2.post),
    se_bd = fit$stdev.unscaled[, "is_bd_psychosis"] * sqrt(fit$s2.post),
    df_total = fit$df.total, residual_sd = fit$sigma)
  model_info[[region]] <- data.frame(region = region, n = nrow(design), p = ncol(design),
    df_residual = fit$df.residual[1], df_prior = fit$df.prior, s2_prior = fit$s2.prior)
  message("Verified unchanged primary model: ", region)
}
write_tsv <- function(x, file) write.table(x, file.path(out, file), sep = "\t", quote = FALSE, row.names = FALSE)
write_tsv(do.call(rbind, uncertainty), "primary_gene_uncertainty.tsv")
write_tsv(do.call(rbind, checks), "primary_fit_verification.tsv")
write_tsv(do.call(rbind, model_info), "primary_model_info.tsv")
writeLines(capture.output(sessionInfo()), file.path(out, "primary_refit_sessionInfo.txt"))
