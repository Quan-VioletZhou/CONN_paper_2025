# =============================================================================
# followup_per_network_cross_Tier2.R
# =============================================================================
# CROSS-SECTIONAL per-network follow-up on the YA + W1 sample (Tier-2 outlier
# removed master sheet). Mirrors followup_per_network_Tier2.R but for the
# cross-sectional question.
#
# Sample: cohort ∈ {YA, W1}, Subgroup == 1, exclude == 0  ->  n = 178
# Model : within_N ~ mean_FD + Btwn_cAge      (one row per subject, lm)
# Bonferroni: α = 0.05 / 8 = 0.00625
#
# Outputs:
#   network_fits_cross                       — named list of fitted lm objects
#   followup_per_network_cross_Tier2.csv     — tidy results table
#   followup_per_network_cross_Tier2.png     — forest plot of Btwn_cAge β
# =============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)

# ---- USER SETTINGS ----------------------------------------------------------
MS  <- "CONNBx_Mastersheet_2026_6.5_final_outlier_removed.csv"
OUT <- "models_Tier2"
dir.create(OUT, showWarnings = FALSE)

NETS       <- c("DMN","SMN","VIS","SAL","DAN","FP","LIM","CRB")
ALPHA_BONF <- 0.05 / length(NETS)             # 0.00625

# ---- 1. Load and filter -----------------------------------------------------
df_cross <- read.csv(MS) %>%
  rename(Sub_Num   = SubNum,
         Age       = Age_DuringParticipation,
         Btwn_cAge = between_cAge,
         With_cAge = within_cAge) %>%
  filter(cohort %in% c("YA", "W1"),     # cross-sectional: YA + W1 only
         Subgroup == 1,                  # HC only
         exclude == 0)                   # motion-clean

cat(sprintf("Sample: %d subjects\n", nrow(df_cross)))
cat("  by cohort: "); print(table(df_cross$cohort))

# ---- 2. Loop through 8 networks; fit lm; store both model and summary -------
network_fits_cross <- list()              # full lm model objects
results_rows       <- list()              # tidy summary rows

for (net in NETS) {
  out_col <- paste0(net, "_within")
  d <- df_cross %>%
    select(Subject, Sub_Num, cohort, mean_FD, Btwn_cAge,
           value = all_of(out_col)) %>%
    filter(!is.na(value), !is.na(Btwn_cAge), !is.na(mean_FD))

  fit <- lm(value ~ mean_FD + Btwn_cAge, data = d)
  network_fits_cross[[net]] <- fit

  s <- summary(fit)$coefficients
  results_rows[[net]] <- data.frame(
    network   = net,
    term      = "Btwn_cAge",
    beta      = s["Btwn_cAge", "Estimate"],
    SE        = s["Btwn_cAge", "Std. Error"],
    df        = fit$df.residual,
    t         = s["Btwn_cAge", "t value"],
    p         = s["Btwn_cAge", "Pr(>|t|)"],
    n         = nobs(fit),
    bonf_pass = s["Btwn_cAge", "Pr(>|t|)"] < ALPHA_BONF
  )

  cat(sprintf("  %-5s  β = %+.5f   SE = %.5f   t = %5.2f   p = %.4f\n",
              net,
              s["Btwn_cAge","Estimate"],
              s["Btwn_cAge","Std. Error"],
              s["Btwn_cAge","t value"],
              s["Btwn_cAge","Pr(>|t|)"]))
}

results_cross <- bind_rows(results_rows) %>%
  mutate(flag = case_when(
    bonf_pass ~ "*** Bonferroni",
    p < .001  ~ "***",
    p < .01   ~ "**",
    p < .05   ~ "*",
    TRUE      ~ "ns"))

# ---- 3. Save results --------------------------------------------------------
write.csv(results_cross,
          file.path(OUT, "followup_per_network_cross_Tier2.csv"),
          row.names = FALSE)
cat(sprintf("\nTidy results -> %s\n",
            file.path(OUT, "followup_per_network_cross_Tier2.csv")))

saveRDS(network_fits_cross,
        file.path(OUT, "network_fits_cross_Tier2.rds"))
cat(sprintf("Model objects -> %s\n",
            file.path(OUT, "network_fits_cross_Tier2.rds")))

# ---- 4. Forest plot ---------------------------------------------------------
plot_df <- results_cross %>%
  mutate(network = factor(network, levels = NETS),
         lo = beta - 1.96*SE,
         hi = beta + 1.96*SE,
         color_grp = case_when(
           bonf_pass ~ "Bonferroni",
           p < .05   ~ "p < .05",
           TRUE      ~ "ns"),
         color_grp = factor(color_grp,
                            levels = c("Bonferroni","p < .05","ns")))

p <- ggplot(plot_df,
            aes(x = beta, y = network, color = color_grp)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.9) +
  scale_color_manual(values = c(Bonferroni = "#cd3e4e",
                                `p < .05`   = "#e69422",
                                ns          = "grey60")) +
  scale_y_discrete(limits = rev) +
  labs(x = "β (Btwn_cAge, 95% CI) — change per year of age",
       y = NULL, color = NULL,
       title    = "Cross-sectional per-network follow-up — YA + W1 (n = 178)",
       subtitle = sprintf("Bonferroni-corrected across %d networks (α = %.4f)",
                          length(NETS), ALPHA_BONF)) +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom",
        plot.title    = element_text(face = "bold"))

ggsave(file.path(OUT, "followup_per_network_cross_Tier2.png"),
       p, width = 7.5, height = 5, dpi = 150)
cat(sprintf("Forest plot   -> %s\n\n",
            file.path(OUT, "followup_per_network_cross_Tier2.png")))

# ---- 5. Print results table -------------------------------------------------
cat("===== Results =====\n")
print(results_cross %>%
        mutate(beta = round(beta, 5), SE = round(SE, 5),
               t = round(t, 2), p = signif(p, 3)) %>%
        select(network, beta, SE, df, t, p, flag),
      row.names = FALSE)

cat(sprintf("\nBonferroni α = %.5f  (8 networks)\n", ALPHA_BONF))

# ---- 6. Quick inspect helper ------------------------------------------------
# To revisit any individual model later:
#   summary(network_fits_cross[["SAL"]])
#   confint(network_fits_cross[["SAL"]])
#   plot(network_fits_cross[["SAL"]])
