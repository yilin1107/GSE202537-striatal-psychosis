suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(MuSiC)
})

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/40_run_brain_immune_music.R"
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)

processed_dir <- file.path(root, "data", "processed")
table_dir <- file.path(root, "results", "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

counts <- readMM(file.path(processed_dir, "brain_immune_music_reference_counts.mtx"))
features <- read.delim(file.path(processed_dir, "brain_immune_music_reference_features.tsv"), check.names = FALSE)
barcodes <- read.delim(file.path(processed_dir, "brain_immune_music_reference_barcodes.tsv"), header = FALSE, check.names = FALSE)[[1]]
metadata <- read.delim(file.path(processed_dir, "brain_immune_music_reference_metadata.tsv"), check.names = FALSE)

rownames(counts) <- make.unique(features$gene_symbol)
colnames(counts) <- barcodes
metadata <- metadata[match(colnames(counts), metadata$cell), ]

sce <- SingleCellExperiment(
  assays = list(counts = counts),
  colData = metadata
)

bulk <- read.delim(file.path(processed_dir, "gse202537_gene_cpm_for_iobr.tsv"), check.names = FALSE)
rownames(bulk) <- bulk[[1]]
bulk[[1]] <- NULL
bulk <- as.matrix(bulk)
storage.mode(bulk) <- "numeric"

common_genes <- intersect(rownames(bulk), rownames(sce))
bulk <- bulk[common_genes, , drop = FALSE]
sce <- sce[common_genes, ]

set.seed(20260706)
music_res <- music_prop(
  bulk.mtx = bulk,
  sc.sce = sce,
  clusters = "music_cell_type",
  samples = "music_sample_id",
  verbose = FALSE,
  normalize = FALSE
)

weighted <- as.data.frame(music_res$Est.prop.weighted)
weighted <- cbind(sample_key = rownames(weighted), weighted)
names(weighted) <- gsub("[[:space:]]+", "_", names(weighted))
write.table(
  weighted,
  file.path(table_dir, "npj_figure4_brain_immune_music_proportions_wide.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

unweighted <- as.data.frame(music_res$Est.prop.allgene)
unweighted <- cbind(sample_key = rownames(unweighted), unweighted)
names(unweighted) <- gsub("[[:space:]]+", "_", names(unweighted))
write.table(
  unweighted,
  file.path(table_dir, "npj_figure4_brain_immune_music_proportions_allgene.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

message("Wrote MuSiC proportions using ", length(common_genes), " common genes and ", ncol(sce), " reference cells.")
