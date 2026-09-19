suppressPackageStartupMessages({
  library(IOBR)
})

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/35_run_iobr_cibersort.R"
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
input_cpm <- file.path(root, "data", "processed", "gse202537_gene_cpm_for_iobr.tsv")
output_dir <- file.path(root, "results", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_wide <- file.path(output_dir, "npj_figure2_iobr_cibersort_wide.tsv")

expr <- read.delim(input_cpm, check.names = FALSE, stringsAsFactors = FALSE)
rownames(expr) <- expr[[1]]
expr[[1]] <- NULL
expr <- as.data.frame(expr, check.names = FALSE)

set.seed(20260706)
res <- deconvo_tme(
  eset = expr,
  method = "cibersort",
  arrays = FALSE,
  perm = 100,
  absolute.mode = FALSE
)

write.table(res, output_wide, sep = "\t", quote = FALSE, row.names = FALSE)
message("Wrote ", output_wide)
