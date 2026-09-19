#!/usr/bin/env Rscript
# 66_revision_deconvolution_alternatives.R
# MDPI Biology revision, analysis A6 (reference-dependence of the putamen OPC / oligodendrocyte-lineage result):
#   (1) CIBERSORT (IOBR implementation, nu-SVR) with the HPA/Siletti 7-class signature matrix used in the manuscript
#   (2) MuSiC with an independent multi-donor human striatal (NAc) single-nucleus reference: Tran et al. 2021, Neuron
#       (8 donors; SingleCellExperiment object SCE_NAc-n8_tran-etal.rda from
#        https://libd-snrnaseq-pilot.s3.us-east-2.amazonaws.com/SCE_NAc-n8_tran-etal.rda  -> save to data/raw/)
# Inputs  : data/processed/gse202537_gene_cpm_for_iobr.tsv (bulk CPM), results/tables/revision_hpa_siletti_signature_matrix.tsv
#           data/raw/SCE_NAc-n8_tran-etal.rda (download manually; ~ several hundred MB)
# Outputs : results/tables/revision_R_A6_cibersort_hpa_proportions.tsv
#           results/tables/revision_R_A6_music_tran_nac_proportions.tsv (+ _allgene, _reference_summary)
# Run from the project root:  Rscript scripts/66_revision_deconvolution_alternatives.R
suppressPackageStartupMessages({
  library(Matrix)
})
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/66_revision_deconvolution_alternatives.R"
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
processed_dir <- file.path(root, "data", "processed")
raw_dir <- file.path(root, "data", "raw")
table_dir <- file.path(root, "results", "tables")

bulk <- read.delim(file.path(processed_dir, "gse202537_gene_cpm_for_iobr.tsv"), check.names = FALSE)
rownames(bulk) <- toupper(bulk[[1]]); bulk[[1]] <- NULL
bulk <- as.matrix(bulk); storage.mode(bulk) <- "numeric"

# ------------------------------------------------------------------ (1) CIBERSORT with the HPA/Siletti signature matrix
sig <- read.delim(file.path(table_dir, "revision_hpa_siletti_signature_matrix.tsv"), check.names = FALSE)
rownames(sig) <- toupper(sig[[1]]); sig[[1]] <- NULL
common <- intersect(rownames(sig), rownames(bulk))
message("CIBERSORT: ", length(common), " signature genes present in bulk")
if (requireNamespace("IOBR", quietly = TRUE)) {
  set.seed(20260919)
  cib <- IOBR::CIBERSORT(sig_matrix = as.data.frame(sig[common, , drop = FALSE]),
                         mixture_file = as.data.frame(bulk[common, , drop = FALSE]),
                         perm = 100, QN = FALSE, absolute = FALSE)
  cib <- as.data.frame(cib)
  cib <- cbind(sample_key = rownames(cib), cib)
  write.table(cib, file.path(table_dir, "revision_R_A6_cibersort_hpa_proportions.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  message("Wrote revision_R_A6_cibersort_hpa_proportions.tsv")
} else {
  message("IOBR not installed; skipping CIBERSORT step.")
}

# ------------------------------------------------------------------ (2) MuSiC with Tran et al. 2021 NAc snRNA-seq reference
sce_file <- file.path(raw_dir, "SCE_NAc-n8_tran-etal.rda")
if (file.exists(sce_file) && requireNamespace("MuSiC", quietly = TRUE) && requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  suppressPackageStartupMessages({ library(SingleCellExperiment); library(MuSiC) })
  env <- new.env(); load(sce_file, envir = env)
  objs <- ls(env)
  sce <- NULL
  for (o in objs) if (inherits(env[[o]], "SingleCellExperiment")) { sce <- env[[o]]; message("Using object: ", o) }
  stopifnot(!is.null(sce))
  cd <- as.data.frame(colData(sce))
  ct_col <- intersect(c("cellType", "cellType.final", "cellType.broad", "cell_type", "prelimCluster"), colnames(cd))[1]
  donor_col <- intersect(c("donor", "Sample", "sample", "sampleID", "donor_id", "brnum"), colnames(cd))[1]
  stopifnot(!is.na(ct_col), !is.na(donor_col))
  message("cell-type column: ", ct_col, " ; donor column: ", donor_col)
  ct <- as.character(cd[[ct_col]])
  # collapse Tran et al. fine labels into the seven broad classes used in the manuscript
  cls <- rep(NA_character_, length(ct))
  cls[grepl("^MSN", ct, ignore.case = TRUE)] <- "MSN"
  cls[grepl("^Inhib|Interneuron|^GABA", ct, ignore.case = TRUE)] <- "Interneuron"
  cls[grepl("^Astro", ct, ignore.case = TRUE)] <- "Astrocyte"
  cls[grepl("^Oligo", ct, ignore.case = TRUE)] <- "Oligodendrocyte"
  cls[grepl("^OPC|^COP", ct, ignore.case = TRUE)] <- "OPC"
  cls[grepl("^Micro|^Macro", ct, ignore.case = TRUE)] <- "Microglia"
  cls[grepl("^Endo|^Mural|^Peri|Vasc", ct, ignore.case = TRUE)] <- "Vascular"
  keep <- !is.na(cls)
  message("Nuclei kept: ", sum(keep), " of ", length(cls), " ; dropped labels: ", paste(unique(ct[!keep]), collapse = ", "))
  sce <- sce[, keep]; sce$music_class <- cls[keep]; sce$music_donor <- as.character(cd[[donor_col]][keep])
  ref_summary <- as.data.frame(table(sce$music_class, sce$music_donor))
  names(ref_summary) <- c("class", "donor", "n_nuclei")
  write.table(ref_summary, file.path(table_dir, "revision_R_A6_music_tran_nac_reference_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  # gene symbols
  rn <- if ("gene_name" %in% colnames(rowData(sce))) rowData(sce)$gene_name else if ("Symbol" %in% colnames(rowData(sce))) rowData(sce)$Symbol else rownames(sce)
  rownames(sce) <- make.unique(toupper(as.character(rn)))
  if (!"counts" %in% assayNames(sce)) stop("SCE has no 'counts' assay")
  common <- intersect(rownames(sce), rownames(bulk))
  message("MuSiC: ", length(common), " common genes")
  set.seed(20260919)
  res <- music_prop(bulk.mtx = bulk[common, , drop = FALSE], sc.sce = sce[common, ], clusters = "music_class",
                    samples = "music_donor", verbose = FALSE)
  w <- as.data.frame(res$Est.prop.weighted); w <- cbind(sample_key = rownames(w), w)
  write.table(w, file.path(table_dir, "revision_R_A6_music_tran_nac_proportions.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  a <- as.data.frame(res$Est.prop.allgene); a <- cbind(sample_key = rownames(a), a)
  write.table(a, file.path(table_dir, "revision_R_A6_music_tran_nac_proportions_allgene.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  message("Wrote revision_R_A6_music_tran_nac_proportions.tsv")
} else {
  message("Skipping MuSiC: need data/raw/SCE_NAc-n8_tran-etal.rda and packages MuSiC + SingleCellExperiment.")
}
writeLines(capture.output(sessionInfo()), file.path(table_dir, "revision_R_A6_sessionInfo.txt"))
