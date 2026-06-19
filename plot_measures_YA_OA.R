# =============================================================================
# plot_measures_YA_OA.R
# =============================================================================
# Multi-panel boxplots of every FC measure, comparing Young (YA) vs Old (W1+W2),
# HC only (Subgroup == 1) and motion-clean (exclude == 0).
#
# Outliers (|z| > 3 SD) are computed WITHIN each age group
# (so a YA outlier is "unusual relative to other YA", not relative to OA).
#
# Output:
#   plots_YA_OA/all_measures_YA_OA.pdf    -- multi-panel boxplot PDF
#   plots_YA_OA/outliers_by_group.csv     -- which Subject is an outlier on what
#   plots_YA_OA/exclude_summary_z3.csv    -- per-subject summary (n_flagged etc.)
# =============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
if (!requireNamespace("ggrepel", quietly = TRUE)) install.packages("ggrepel")
library(ggrepel)

# ---- USER SETTINGS ----------------------------------------------------------
MS    <- "CONNBx_Mastersheet_2026_6.3_final.csv"     # adjust to your path
OUT   <- "plots_YA_OA"
Z_CUT <- 3
dir.create(OUT, showWarnings = FALSE)

NETS <- c("DMN","SMN","VIS","SAL","DAN","FP","LIM","CRB")
MEASURES <- c(
  paste0(NETS, "_within"),
  paste0(NETS, "_between"),
  paste0(NETS, "_seg"),
  c("DMN_DAN","DMN_FP","DMN_SAL","DAN_FP","DAN_SAL","FP_SAL"),
  c("global_within","global_between","global_seg")
)   # 33 measures total

# ---- 1. Load and filter (HC + motion-clean + no MCI cohort) ----------------
df <- read.csv(MS) %>%
  filter(cohort %in% c("YA","W1","W2"),
         Subgroup == 1,
         exclude == 0) %>%
  mutate(age_group = factor(ifelse(cohort == "YA", "Young", "Old"),
                            levels = c("Young", "Old")))

cat(sprintf("Sample: %d rows  (Young = %d, Old = %d)\n",
            nrow(df), sum(df$age_group=="Young"), sum(df$age_group=="Old")))
cat(sprintf("Unique people: %d\n\n", length(unique(df$SubNum))))

# ---- 2. Build one ggplot panel per measure ----------------------------------
make_panel <- function(measure_name) {
  d <- df %>%
    select(Subject, age_group, all_of(measure_name)) %>%
    rename(value = !!measure_name) %>%
    filter(!is.na(value)) %>%
    group_by(age_group) %>%
    mutate(z = (value - mean(value)) / sd(value),
           outlier = abs(z) > Z_CUT) %>%
    ungroup()

  ggplot(d, aes(x = age_group, y = value, color = age_group)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, alpha = 0.30,
                 fill = NA, lwd = 0.5) +
    geom_jitter(width = 0.18, size = 0.9, alpha = 0.55) +
    geom_point(data = d %>% filter(outlier),
               size = 2.4, color = "red", shape = 1, stroke = 0.8) +
    geom_text_repel(
      data = d %>% filter(outlier),
      aes(label = Subject), color = "firebrick", size = 2.2,
      max.overlaps = Inf, segment.size = 0.2, segment.alpha = 0.5,
      box.padding = 0.3) +
    scale_color_manual(values = c(Young = "#4C72B0", Old = "#DD8452")) +
    labs(title = measure_name, x = NULL, y = NULL) +
    theme_classic(base_size = 9) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", size = 9, hjust = 0.5),
          axis.text  = element_text(size = 7))
}

panels <- lapply(MEASURES, make_panel)

# ---- 3. Assemble multi-panel PDF (5 cols x 7 rows) --------------------------
combined <- wrap_plots(panels, ncol = 5) +
  plot_annotation(
    title    = "All FC measures — Young (YA) vs Old (W1+W2 HC)",
    subtitle = sprintf("Red = outlier within group (|z| > %g)", Z_CUT),
    theme = theme(plot.title    = element_text(face = "bold", size = 12),
                  plot.subtitle = element_text(size = 9, color = "grey30")))

pdf_path <- file.path(OUT, "all_measures_YA_OA.pdf")
ggsave(pdf_path, combined, width = 18, height = 24, units = "in")
cat(sprintf("Multi-panel PDF saved -> %s\n", pdf_path))

# ---- 4. Flat outlier table (per-measure × subject) --------------------------
outliers_all <- do.call(rbind, lapply(MEASURES, function(m) {
  df %>%
    select(Subject, cohort, age_group, all_of(m)) %>%
    rename(value = !!m) %>%
    filter(!is.na(value)) %>%
    group_by(age_group) %>%
    mutate(z = (value - mean(value)) / sd(value),
           outlier = abs(z) > Z_CUT) %>%
    ungroup() %>%
    filter(outlier) %>%
    mutate(measure = m) %>%
    select(Subject, cohort, age_group, measure, value, z)
}))

write.csv(outliers_all, file.path(OUT, "outliers_by_group.csv"), row.names = FALSE)
cat(sprintf("Outlier table saved   -> %s\n", file.path(OUT, "outliers_by_group.csv")))
cat(sprintf("%d total (Subject x measure) outlier entries.\n", nrow(outliers_all)))

# ---- 5. Per-subject summary -------------------------------------------------
exclude_summary <- outliers_all %>%
  group_by(Subject, cohort, age_group) %>%
  summarise(n_flagged = n(),
            max_z     = max(abs(z)),
            measures  = paste(sort(measure), collapse = ", "),
            .groups   = "drop") %>%
  arrange(age_group, desc(n_flagged), desc(max_z))
write.csv(exclude_summary, file.path(OUT, "exclude_summary_z3.csv"), row.names = FALSE)
cat(sprintf("Per-subject summary   -> %s\n",
            file.path(OUT, "exclude_summary_z3.csv")))

# Severity tiers
cat(sprintf("\n--- Severity tier ---\n"))
cat(sprintf("  1 measure flagged    : %d subjects\n", sum(exclude_summary$n_flagged == 1)))
cat(sprintf("  2 measures flagged   : %d subjects\n", sum(exclude_summary$n_flagged == 2)))
cat(sprintf("  3+ measures flagged  : %d subjects\n", sum(exclude_summary$n_flagged >= 3)))

cat("\nSubjects flagged on >=3 measures (recommended for exclusion):\n")
print(exclude_summary %>% filter(n_flagged >= 3), row.names = FALSE)
