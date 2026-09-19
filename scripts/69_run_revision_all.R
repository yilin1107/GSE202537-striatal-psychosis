# 69_run_revision_all.R
# Driver for the R part of the sensitivity analyses (Rscript scripts/69_run_revision_all.R).
# It sets the working directory to the project root, then sources steps 68 (packages + Tran 2021 reference),
# 65 (limma-voom / camera / run sensitivity) and 66 (CIBERSORT-HPA + MuSiC-Tran) in order, and finally runs the
# Python comparison (67) if python is on PATH.  Console output is also written to
# results/tables/revision_R_run.log.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/", mustWork = FALSE)
  default_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
} else {
  default_root <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
root <- Sys.getenv("GSE202537_PROJECT_ROOT", unset = default_root)
setwd(root)
dir.create("results/tables", showWarnings = FALSE, recursive = TRUE)
logfile <- "results/tables/revision_R_run.log"
con <- file(logfile, open = "wt")
sink(con, split = TRUE)
sink(con, type = "message")
cat("===== revision R run started", format(Sys.time()), "\n")
cat("R:", R.version.string, "| wd:", getwd(), "\n")
flush(con)

run_step <- function(script) {
  cat("\n=====", script, "started", format(Sys.time()), "\n")
  flush(con)
  t0 <- Sys.time()
  res <- try(source(script, local = new.env(), echo = FALSE), silent = TRUE)
  if (inherits(res, "try-error")) cat("ERROR in", script, ":", conditionMessage(attr(res, "condition")), "\n")
  cat("=====", script, "finished", format(Sys.time()), "elapsed", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
  flush(con)
  invisible(!inherits(res, "try-error"))
}

ok68 <- run_step("scripts/68_revision_install_and_download.R")
ok65 <- run_step("scripts/65_revision_limma_voom_camera.R")
ok66 <- run_step("scripts/66_revision_deconvolution_alternatives.R")

py <- Sys.which("python")
if (nzchar(py)) {
  cat("\n===== scripts/67_revision_compare_python_vs_R.py (", py, ")\n")
  out <- try(system2(py, "scripts/67_revision_compare_python_vs_R.py", stdout = TRUE, stderr = TRUE), silent = TRUE)
  if (inherits(out, "try-error")) cat("python step failed\n") else cat(out, sep = "\n")
} else {
  cat("\npython not found on PATH; run scripts/67_revision_compare_python_vs_R.py manually later.\n")
}
cat("\n===== finished", format(Sys.time()), "| 68:", ok68, "65:", ok65, "66:", ok66, "\n")
flush(con)
sink(type = "message")
sink()
close(con)
cat("All done. Log: results/tables/revision_R_run.log\n")
