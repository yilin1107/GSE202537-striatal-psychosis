#!/usr/bin/env Rscript
# 65_revision_limma_voom_camera.R
# MDPI Biology revision, reference implementation (R/Bioconductor) of analyses A1-A3:
#   A1  limma-voom region-specific diagnosis models (SCZ, BD-psychosis vs control; same covariates as the OLS model)
#   A3  correlation-aware Hallmark tests: camera (inter.gene.cor = 0.01 and estimated), fry
#   A2  sequencing-run sensitivity: run as fixed effect, duplicateCorrelation (run as random block),
#       RUVr residual factors (k = 1-3), svaseq surrogate variables
# Inputs  : data/processed/gse202537_symbol_counts_filtered.tsv   (20,810 symbol-level raw counts, manuscript filter)
#           data/processed/gse202537_sample_metadata.csv, data/processed/gse202537_expression_qc.csv
#           data/raw/MSigDB_Hallmark_2020.gmt, results/tables/npj_figure1_hallmark_definitions.tsv
# Outputs : results/tables/revision_R_*.tsv (+ results/tables/revision_R_sessionInfo.txt)
# Run from the project root:  Rscript scripts/65_revision_limma_voom_camera.R
suppressPackageStartupMessages({
  library(edgeR)
  library(limma)
})
has_ruv <- requireNamespace("RUVSeq", quietly = TRUE)
has_sva <- requireNamespace("sva", quietly = TRUE)

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/65_revision_limma_voom_camera.R"
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
processed_dir <- file.path(root, "data", "processed")
raw_dir <- file.path(root, "data", "raw")
table_dir <- file.path(root, "results", "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------- data
counts <- as.matrix(read.delim(file.path(processed_dir, "gse202537_symbol_counts_filtered.tsv"), row.names = 1, check.names = FALSE))
meta <- read.csv(file.path(processed_dir, "gse202537_sample_metadata.csv"), check.names = FALSE)
qc <- read.csv(file.path(processed_dir, "gse202537_expression_qc.csv"), check.names = FALSE)
meta <- merge(meta, qc[, c("sample_key", "count_depth_million")], by = "sample_key", all.x = TRUE)
meta <- meta[match(colnames(counts), meta$sample_key), ]
stopifnot(all(meta$sample_key == colnames(counts)))
meta$sex_male <- as.numeric(meta$gender == "Male")
meta$tod_sin <- sin(2 * pi * meta$corrected_tod_24h / 24)
meta$tod_cos <- cos(2 * pi * meta$corrected_tod_24h / 24)
meta$is_scz <- as.numeric(meta$group == "SCZ")
meta$is_bd_psychosis <- as.numeric(meta$group == "BD with psychosis")
regions <- c("NAc", "Caudate", "Putamen")

make_design <- function(m, run_fixed = FALSE, extra = NULL) {
  z <- function(x) { s <- sd(x); if (is.finite(s) && s > 0) (x - mean(x)) / s else rep(0, length(x)) }
  d <- data.frame(
    intercept = 1,
    is_scz = m$is_scz,
    is_bd_psychosis = m$is_bd_psychosis,
    age_z = z(m$age), sex_male = m$sex_male, pmi_z = z(m$pmi), rin_z = z(m$rin), ph_z = z(m$ph),
    count_depth_million_z = z(m$count_depth_million), tod_sin = m$tod_sin, tod_cos = m$tod_cos
  )
  if (run_fixed) {
    run <- factor(m$sequence_id)
    rm <- model.matrix(~run)[, -1, drop = FALSE]
    colnames(rm) <- paste0("run_", levels(run)[-1])
    d <- cbind(d, rm)
  }
  if (!is.null(extra)) d <- cbind(d, extra)
  X <- as.matrix(d)
  rownames(X) <- m$sample_key
  X
}

# Hallmark sets (same parsing as scripts/32: upper-case symbols)
gmt_lines <- readLines(file.path(raw_dir, "MSigDB_Hallmark_2020.gmt"))
gmt <- lapply(strsplit(gmt_lines, "\t"), function(p) unique(toupper(trimws(p[-(1:2)]))))
names(gmt) <- trimws(sapply(strsplit(gmt_lines, "\t"), `[`, 1))
defs <- read.delim(file.path(table_dir, "npj_figure1_hallmark_definitions.tsv"), check.names = FALSE)
hallmark_index <- lapply(seq_len(nrow(defs)), function(i) which(rownames(counts) %in% gmt[[defs$hallmark[i]]]))
names(hallmark_index) <- defs$hallmark_id

# ---------------------------------------------------------------- TMM (global, as one DGEList) then per-region voom
dge <- DGEList(counts = counts)
dge <- calcNormFactors(dge, method = "TMM")
write.table(data.frame(sample_key = colnames(dge), lib_size = dge$samples$lib.size, norm_factor = dge$samples$norm.factors),
            file.path(table_dir, "revision_R_A1_tmm_factors.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

tt_frame <- function(fit, coef, region, contrast, variant = "primary") {
  tt <- topTable(fit, coef = coef, number = Inf, sort.by = "none")
  data.frame(variant = variant, gene = rownames(tt), brain_region = region, contrast = contrast,
             AveExpr = tt$AveExpr, logFC = tt$logFC, t = tt$t, P.Value = tt$P.Value, adj.P.Val = tt$adj.P.Val,
             stringsAsFactors = FALSE)
}
gs_frame <- function(v, design, coef, region, contrast, variant = "primary") {
  cam <- camera(v, hallmark_index, design = design, contrast = coef, inter.gene.cor = 0.01, sort = FALSE)
  cam_est <- camera(v, hallmark_index, design = design, contrast = coef, inter.gene.cor = NA, sort = FALSE)
  fr <- fry(v, hallmark_index, design = design, contrast = coef, sort = FALSE)
  data.frame(variant = variant, brain_region = region, contrast = contrast, set = rownames(cam),
             NGenes = cam$NGenes, camera_direction = cam$Direction, camera_p = cam$PValue, camera_fdr = cam$FDR,
             estimated_intergene_cor = cam_est$Correlation, camera_estcor_p = cam_est$PValue, camera_estcor_fdr = cam_est$FDR,
             fry_direction = fr$Direction, fry_p = fr$PValue, fry_fdr = fr$FDR,
             fry_mixed_p = fr$PValue.Mixed, fry_mixed_fdr = fr$FDR.Mixed, stringsAsFactors = FALSE)
}

gene_out <- list(); gs_out <- list(); info_out <- list()
for (region in regions) {
  idx <- which(meta$brain_region == region)
  m <- meta[idx, ]
  y <- dge[, idx]                       # keeps global TMM factors and library sizes
  design <- make_design(m)
  v <- voom(y, design)
  fit <- eBayes(lmFit(v, design))
  for (cc in c("is_scz", "is_bd_psychosis")) {
    contrast <- if (cc == "is_scz") "SCZ_vs_Control" else "BD_psychosis_vs_Control"
    gene_out[[length(gene_out) + 1]] <- tt_frame(fit, cc, region, contrast)
    gs_out[[length(gs_out) + 1]] <- gs_frame(v, design, cc, region, contrast)
  }
  info_out[[length(info_out) + 1]] <- data.frame(variant = "primary", region = region, n = ncol(y), p = ncol(design),
                                                 df_residual = fit$df.residual[1], df_prior = fit$df.prior, s2_prior = fit$s2.prior)

  # ---- A2: run as fixed effect
  design_run <- make_design(m, run_fixed = TRUE)
  v_run <- voom(y, design_run)
  fit_run <- eBayes(lmFit(v_run, design_run))
  for (cc in c("is_scz", "is_bd_psychosis")) {
    contrast <- if (cc == "is_scz") "SCZ_vs_Control" else "BD_psychosis_vs_Control"
    gene_out[[length(gene_out) + 1]] <- tt_frame(fit_run, cc, region, contrast, "run_fixed")
    gs_out[[length(gs_out) + 1]] <- gs_frame(v_run, design_run, cc, region, contrast, "run_fixed")
  }
  info_out[[length(info_out) + 1]] <- data.frame(variant = "run_fixed", region = region, n = ncol(y), p = ncol(design_run),
                                                 df_residual = fit_run$df.residual[1], df_prior = fit_run$df.prior, s2_prior = fit_run$s2.prior)

  # ---- A2: duplicateCorrelation with run as random block (two-pass, as recommended in the limma guide)
  block <- factor(m$sequence_id)
  corfit <- duplicateCorrelation(v, design, block = block)
  v_dc <- voom(y, design, block = block, correlation = corfit$consensus)
  corfit2 <- duplicateCorrelation(v_dc, design, block = block)
  fit_dc <- eBayes(lmFit(v_dc, design, block = block, correlation = corfit2$consensus))
  for (cc in c("is_scz", "is_bd_psychosis")) {
    contrast <- if (cc == "is_scz") "SCZ_vs_Control" else "BD_psychosis_vs_Control"
    gene_out[[length(gene_out) + 1]] <- tt_frame(fit_dc, cc, region, contrast, "dupcor")
    gs_out[[length(gs_out) + 1]] <- gs_frame(v_dc, design, cc, region, contrast, "dupcor")
  }
  info_out[[length(info_out) + 1]] <- data.frame(variant = "dupcor", region = region, n = ncol(y), p = ncol(design),
                                                 df_residual = fit_dc$df.residual[1], df_prior = fit_dc$df.prior, s2_prior = fit_dc$s2.prior,
                                                 consensus_correlation = corfit2$consensus)

  # ---- A2: RUVr residual factors (k = 1..3) from an edgeR first-pass fit
  if (has_ruv) {
    y_glm <- estimateGLMCommonDisp(y, design)
    y_glm <- estimateGLMTagwiseDisp(y_glm, design)
    fit_glm <- glmFit(y_glm, design)
    res <- residuals(fit_glm, type = "deviance")
    for (k in 1:3) {
      ruv <- RUVSeq::RUVr(round(as.matrix(y$counts)), rownames(y), k = k, res)
      W <- ruv$W; colnames(W) <- paste0("ruv", seq_len(k))
      design_ruv <- make_design(m, extra = W)
      v_ruv <- voom(y, design_ruv)
      fit_ruv <- eBayes(lmFit(v_ruv, design_ruv))
      for (cc in c("is_scz", "is_bd_psychosis")) {
        contrast <- if (cc == "is_scz") "SCZ_vs_Control" else "BD_psychosis_vs_Control"
        gene_out[[length(gene_out) + 1]] <- tt_frame(fit_ruv, cc, region, contrast, paste0("ruvr_k", k))
        gs_out[[length(gs_out) + 1]] <- gs_frame(v_ruv, design_ruv, cc, region, contrast, paste0("ruvr_k", k))
      }
      info_out[[length(info_out) + 1]] <- data.frame(variant = paste0("ruvr_k", k), region = region, n = ncol(y), p = ncol(design_ruv),
                                                     df_residual = fit_ruv$df.residual[1], df_prior = fit_ruv$df.prior, s2_prior = fit_ruv$s2.prior)
    }
  } else message("RUVSeq not installed: skipping RUVr variants (BiocManager::install('RUVSeq')).")

  # ---- A2: svaseq surrogate variables
  if (has_sva) {
    mod0 <- design[, !(colnames(design) %in% c("is_scz", "is_bd_psychosis")), drop = FALSE]
    sv <- tryCatch(sva::svaseq(as.matrix(y$counts), design, mod0), error = function(e) NULL)
    if (!is.null(sv) && sv$n.sv > 0) {
      SV <- sv$sv; colnames(SV) <- paste0("sv", seq_len(ncol(SV)))
      design_sv <- make_design(m, extra = SV)
      v_sv <- voom(y, design_sv)
      fit_sv <- eBayes(lmFit(v_sv, design_sv))
      for (cc in c("is_scz", "is_bd_psychosis")) {
        contrast <- if (cc == "is_scz") "SCZ_vs_Control" else "BD_psychosis_vs_Control"
        gene_out[[length(gene_out) + 1]] <- tt_frame(fit_sv, cc, region, contrast, paste0("svaseq_n", sv$n.sv))
        gs_out[[length(gs_out) + 1]] <- gs_frame(v_sv, design_sv, cc, region, contrast, paste0("svaseq_n", sv$n.sv))
      }
      info_out[[length(info_out) + 1]] <- data.frame(variant = paste0("svaseq_n", sv$n.sv), region = region, n = ncol(y), p = ncol(design_sv),
                                                     df_residual = fit_sv$df.residual[1], df_prior = fit_sv$df.prior, s2_prior = fit_sv$s2.prior)
    }
  } else message("sva not installed: skipping svaseq variant (BiocManager::install('sva')).")
  message("Finished region ", region)
}

gene_res <- do.call(rbind, gene_out)
write.table(gene_res[gene_res$variant == "primary", -1], file.path(table_dir, "revision_R_A1_voom_gene_results.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(gene_res[gene_res$variant != "primary", ], file.path(table_dir, "revision_R_A2_run_sensitivity_gene_results.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
gs_res <- do.call(rbind, gs_out)
write.table(gs_res[gs_res$variant == "primary", -1], file.path(table_dir, "revision_R_A3_hallmark_camera_fry.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(gs_res, file.path(table_dir, "revision_R_A2_run_sensitivity_hallmark_camera.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
info <- do.call(plyr_rbind <- function(...) { l <- list(...); cols <- unique(unlist(lapply(l, names))); do.call(rbind, lapply(l, function(d) { d[setdiff(cols, names(d))] <- NA; d[cols] })) }, info_out)
write.table(info, file.path(table_dir, "revision_R_A2_model_info.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(table_dir, "revision_R_sessionInfo.txt"))
message("Done. Wrote revision_R_A1/A2/A3 tables to ", table_dir)
