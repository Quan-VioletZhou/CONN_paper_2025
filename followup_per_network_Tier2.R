# =============================================================================
# followup_per_network_Tier2.R
# =============================================================================
# Per-network follow-up linear mixed-effects models on the locked Tier-2 sample.
# Uses the outlier-removed master sheet (6.5) directly, so no extra outlier
# filter is needed in this script.
#
# Sample: cohort ∈ {W1, W2}, Subgroup == 1, exclude == 0
#         (W1 + W2 only — OA longitudinal follow-up)
# Model : within_N ~ mean_FD + Btwn_cAge + With_cAge,  random = ~ 1 | Sub_Num
# Bonferroni: α = 0.05 / 8 = 0.00625
#
# Outputs:
#   network_fits                       — named list of fitted lme objects
#   followup_per_network_Tier2.csv     — tidy results table (β, SE, t, p, flag)
#   followup_per_network_Tier2.png     — forest plot of Btwn_cAge and With_cAge
# =============================================================================

library(dplyr)
library(tidyr)
library(nlme)
library(ggplot2)

# ---- USER SETTINGS ----------------------------------------------------------
MS  <- "CONNBx_Mastersheet_2026_6.5_final_outlier_removed.csv"   # locked sheet
OUT <- "models_Tier2"
dir.create(OUT, showWarnings = FALSE)

NETS       <- c("DMN","SMN","VIS","SAL","DAN","FP","LIM","CRB")
ALPHA_BONF <- 0.05 / length(NETS)                # 0.00625

# ---- 1. Build the analysis dataframe ---------------------------------------
df <- read.csv(MS) %>%
  rename(Sub_Num   = SubNum,
         Age       = Age_DuringParticipation,
         Btwn_cAge = between_cAge,
         With_cAge = within_cAge) %>%
  filter(cohort %in% c("W1","W2"),   # OA only (W1+W2 longitudinal)
         Subgroup == 1,                # HC only
         exclude == 0)                  # motion-clean

cat(sprintf("Sample: %d scans, %d unique subjects\n",
            nrow(df), length(unique(df$Sub_Num))))
cat(sprintf("  by cohort: ")); print(table(df$cohort))

# ---- 2. Robust lme fitter — tries multiple optimizers in order --------------
fit_lme_robust <- function(d) {
  for (opt in c("nlminb", "optim", "nlm")) {
    fit <- tryCatch(
      lme(value ~ mean_FD + Btwn_cAge + With_cAge,
          random  = ~ 1 | Sub_Num,
          method  = "ML",
          data    = d,
          control = lmeControl(opt        = opt,
                                msMaxIter  = 500,
                                maxIter    = 500,
                                tolerance  = 1e-6,
                                niterEM    = 50)),
      error   = function(e) NULL,
      warning = function(w) NULL)
    if (!is.null(fit)) {
      message(sprintf("      [converged with opt='%s']", opt))
      return(fit)
    }
  }
  stop("All optimizers failed for this network.")
}

# ---- 3. Loop through 8 networks; fit lme; store both model and summary ------
network_fits <- list()                # full lme model objects
results_rows <- list()                # tidy summary rows

for (net in NETS) {
  out_col <- paste0(net, "_within")
  cat(sprintf("\n=== Fitting %s ===\n", net))

  d <- df %>%
    select(Subject, Sub_Num, cohort, mean_FD, Btwn_cAge, With_cAge,
           value = all_of(out_col)) %>%
    filter(!is.na(value), !is.na(Btwn_cAge), !is.na(With_cAge),
           !is.na(mean_FD))

  cat(sprintf("    n_obs = %d, n_subjects = %d\n",
              nrow(d), length(unique(d$Sub_Num))))

  fit <- fit_lme_robust(d)
  network_fits[[net]] <- fit

  s <- summary(fit)$tTable
  for (term in c("Btwn_cAge","With_cAge")) {
    results_rows[[paste(net, term)]] <- data.frame(
      network   = net,
      term      = term,
      beta      = s[term,"Value"],
      SE        = s[term,"Std.Error"],
      df        = s[term,"DF"],
      t         = s[term,"t-value"],
      p         = s[term,"p-value"],
      bonf_pass = s[term,"p-value"] < ALPHA_BONF)
  }
  cat(sprintf("    Btwn β=%+.5f p=%.4f   With β=%+.5f p=%.4f\n",
              s["Btwn_cAge","Value"], s["Btwn_cAge","p-value"],
              s["With_cAge","Value"], s["With_cAge","p-value"]))
}

results <- bind_rows(results_rows) %>%
  mutate(flag = case_when(
    bonf_pass     ~ "*** Bonferroni",
    p < .001      ~ "***",
    p < .01       ~ "**",
    p < .05       ~ "*",
    TRUE          ~ "ns"))

# ---- 4. Save results --------------------------------------------------------
write.csv(results, file.path(OUT, "followup_per_network_Tier2.csv"),
          row.names = FALSE)
cat(sprintf("\nTidy results -> %s\n",
            file.path(OUT, "followup_per_network_Tier2.csv")))

saveRDS(network_fits, file.path(OUT, "network_fits_Tier2.rds"))
cat(sprintf("Model objects -> %s\n",
            file.path(OUT, "network_fits_Tier2.rds")))

# ---- 5. Forest plot ---------------------------------------------------------
plot_df <- results %>%
  mutate(network = factor(network, levels = NETS),
         lo = beta - 1.96*SE,
         hi = beta + 1.96*SE,
         color_grp = case_when(
           bonf_pass ~ "Bonferroni",
           p < .05   ~ "p < .05",
           TRUE      ~ "ns"),
         color_grp = factor(color_grp,
                            levels = c("Bonferroni","p < .05","ns")),
         term_label = ifelse(term == "Btwn_cAge",
                             "Cross-sectional (Btwn_cAge)",
                             "Longitudinal (With_cAge)"))

p <- ggplot(plot_df,
            aes(x = beta, y = network, color = color_grp)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.8) +
  facet_wrap(~ term_label, scales = "free_x") +
  scale_color_manual(values = c(Bonferroni = "#cd3e4e",
                                `p < .05`   = "#e69422",
                                ns          = "grey60")) +
  scale_y_discrete(limits = rev) +
  labs(x = "β (per year of age, 95% CI)",
       y = NULL, color = NULL,
       title    = "Per-network follow-up — W1 + W2 HC (Tier 2)",
       subtitle = sprintf("Bonferroni-corrected across %d networks (α = %.4f)",
                          length(NETS), ALPHA_BONF)) +
  theme_classic(base_size = 11) +
  theme(legend.position = "bottom",
        plot.title    = element_text(face = "bold"))

ggsave(file.path(OUT, "followup_per_network_Tier2.png"),
       p, width = 10, height = 5.5, dpi = 150)
cat(sprintf("Forest plot   -> %s\n\n",
            file.path(OUT, "followup_per_network_Tier2.png")))

# ---- 6. Print results table -------------------------------------------------
cat("===== Results =====\n")
print(results %>%
        mutate(beta = round(beta, 5), SE = round(SE, 5),
               t = round(t, 2), p = signif(p, 3)) %>%
        select(network, term, beta, SE, t, p, flag),
      row.names = FALSE)

cat(sprintf("\nBonferroni α = %.5f\n", ALPHA_BONF))

# ---- 7. Quick inspect helper -------------------------------------------------
# To revisit any individual model later:
#   summary(network_fits[["SAL"]])
#   fixef(network_fits[["DMN"]])
#   intervals(network_fits[["FP"]], which = "fixed")
