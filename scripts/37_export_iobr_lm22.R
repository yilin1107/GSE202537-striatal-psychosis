suppressPackageStartupMessages({
  library(IOBR)
})

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "scripts/37_export_iobr_lm22.R"
root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE)
output_path <- file.path(root, "data", "processed", "iobr_lm22_signature_matrix.tsv")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

lm22 <- get("lm22", envir = asNamespace("IOBR"))
lm22 <- data.frame(gene_symbol = rownames(lm22), lm22, check.names = FALSE)
write.table(lm22, output_path, sep = "\t", quote = FALSE, row.names = FALSE)
message("Wrote ", output_path)
