#!/usr/bin/env python3
"""
run_models_connbx_6p3.py
========================
Re-run the within-network FC models on the new CONNBx_Mastersheet_2026_6.3_final.csv:
  - Model 1 (cross-sectional, YA + W1) — Btwn_cAge × network + mean_FD
  - Model 2 (longitudinal, W1 + W2)   — Btwn_cAge × net + With_cAge × net + mean_FD
  - Per-network follow-up for both, Bonferroni-corrected across 8 networks

  MCI subjects are EXCLUDED from all models.
  Subjects with exclude == 1 are EXCLUDED from all models.

Network coding: 8 networks {DMN, SMN, VIS, SAL, DAN, FP, LIM, CRB}.
Sum-to-zero effect coding with VIS as the reference (matches the original paper).

    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/run_models_connbx_6p3.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MS = Path("/home/violetz/CONN_revision/CONNBx_Mastersheet_2026_6.3_final.csv")
OUT_DIR = Path("/home/violetz/CONN_revision/models_connbx_6p3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = ["DMN", "SMN", "VIS", "SAL", "DAN", "FP", "LIM", "CRB"]
REF_NET  = "VIS"                          # reference for sum-to-zero coding
ALPHA_BONF = 0.05 / len(NETWORKS)         # 0.00625


# ============================================================================
# 1. Load, filter, reshape to long format
# ============================================================================
df = pd.read_csv(MS)
df = df.rename(columns={"Subject": "subject_id"})
print(f"Master sheet: {len(df)} rows")
print(f"  cohort counts: {df['cohort'].value_counts().to_dict()}")
print(f"  exclude==1 counts: {df.groupby('cohort')['exclude'].sum().to_dict()}")

# Drop MCI + excluded
df = df[(df["cohort"] != "MCI") & (df["exclude"] == 0)].copy()
print(f"\nAfter excluding MCI + exclude==1: {len(df)} rows")
print(f"  cohort counts: {df['cohort'].value_counts().to_dict()}")

# Pivot to long: one row per (subject, network)
within_cols = [f"{n}_within" for n in NETWORKS]
long = df.melt(
    id_vars=["subject_id", "cohort", "SubNum", "Age_DuringParticipation",
             "Btwn_cAge_orig", "Wthn_cAge_orig", "mean_FD"]
        if "Btwn_cAge_orig" in df.columns else
        ["subject_id", "cohort", "SubNum", "Age_DuringParticipation",
         "between_cAge", "within_cAge", "mean_FD"],
    value_vars=within_cols,
    var_name="net_col", value_name="within"
)
long["network"] = long["net_col"].str.replace("_within", "", regex=False)
long = long.rename(columns={"between_cAge": "Btwn_cAge",
                            "within_cAge":  "With_cAge",
                            "SubNum":       "Sub_Num"})
long = long.dropna(subset=["within", "Btwn_cAge"]).copy()
print(f"\nLong-format rows: {len(long)}  ({long['subject_id'].nunique()} subjects × {long['network'].nunique()} networks)")

# Center mean_FD per cohort group (cross-sectional uses YA+W1; longitudinal uses W1+W2)
long["network"] = pd.Categorical(long["network"], categories=NETWORKS)


# ============================================================================
# 2. Build sum-to-zero ("effect") codes for networks (VIS = reference)
# ============================================================================
nets_nonref = [n for n in NETWORKS if n != REF_NET]
for n in nets_nonref:
    long[f"e_{n}"] = np.where(long["network"] == n, 1.0,
                       np.where(long["network"] == REF_NET, -1.0, 0.0))
effect_cols = [f"e_{n}" for n in nets_nonref]


def fit_mixed(formula, data, label):
    """Fit MixedLM with multi-optimizer fallback; print summary line."""
    last_err = None
    for opt in ("bfgs", "lbfgs", "powell", "cg"):
        try:
            m = smf.mixedlm(formula, data=data, groups=data["Sub_Num"]).fit(method=opt, reml=False)
            if np.isfinite(m.aic):
                return m, opt
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"[{label}] convergence failed: {last_err}")


def report_term(m, term):
    """Return (β, SE, z, p) for a given fixed-effect term."""
    if term not in m.params.index:
        return (np.nan,)*4
    return (m.params[term], m.bse[term], m.tvalues[term], m.pvalues[term])


def stars(p, bonf=False):
    if bonf and p < ALPHA_BONF: return "*** Bonferroni"
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    return "ns"


# ============================================================================
# 3. MODEL 1 — Cross-sectional (YA + W1)
# ============================================================================
print("\n" + "=" * 78)
print("MODEL 1 — CROSS-SECTIONAL (YA + W1)  on CONNBx_6.3_final, no MCI, no excluded")
print("=" * 78)
cross = long[long["cohort"].isin(["YA", "W1"])].copy()
cross["mean_FD_c"] = cross["mean_FD"] - cross["mean_FD"].mean()
n_sub_cross = cross["Sub_Num"].nunique() if cross["Sub_Num"].notna().all() else (
    cross["subject_id"].nunique())
# use subject_id as group (unique across cohorts)
cross["grp"] = cross["subject_id"]
print(f"  n subjects = {cross['grp'].nunique()}, "
      f"rows = {len(cross)}  "
      f"({cross.groupby('cohort')['grp'].nunique().to_dict()})")

fml_cross = ("within ~ Btwn_cAge + mean_FD_c + "
             + " + ".join(effect_cols) + " + "
             + " + ".join([f"Btwn_cAge:{e}" for e in effect_cols]))

m1 = smf.mixedlm(fml_cross, data=cross, groups=cross["grp"]).fit(method="bfgs", reml=False)
print(f"\n  AIC = {m1.aic:.2f}   converged = {m1.converged}")
print("  Key fixed effects:")
for term in ["Intercept", "Btwn_cAge", "mean_FD_c"]:
    b, se, z, p = report_term(m1, term)
    print(f"    {term:<15} β = {b:>10.5f}  SE = {se:>8.5f}  z = {z:>7.2f}  p = {p:>8.5f}   {stars(p)}")


# ============================================================================
# 4. MODEL 2 — Longitudinal (W1 + W2)
# ============================================================================
print("\n" + "=" * 78)
print("MODEL 2 — LONGITUDINAL (W1 + W2)  on CONNBx_6.3_final, no MCI, no excluded")
print("=" * 78)
long_d = long[long["cohort"].isin(["W1", "W2"])].copy()
long_d["mean_FD_c"] = long_d["mean_FD"] - long_d["mean_FD"].mean()
print(f"  n subjects = {long_d['Sub_Num'].nunique()},  n rows = {len(long_d)}  "
      f"({long_d.groupby('cohort')['Sub_Num'].nunique().to_dict()})")

fml_long = ("within ~ Btwn_cAge + With_cAge + mean_FD_c + "
            + " + ".join(effect_cols) + " + "
            + " + ".join([f"Btwn_cAge:{e}" for e in effect_cols]) + " + "
            + " + ".join([f"With_cAge:{e}" for e in effect_cols]))

m2 = smf.mixedlm(fml_long, data=long_d, groups=long_d["Sub_Num"]).fit(method="bfgs", reml=False)
print(f"\n  AIC = {m2.aic:.2f}   converged = {m2.converged}")
print("  Key fixed effects:")
for term in ["Intercept", "Btwn_cAge", "With_cAge", "mean_FD_c"]:
    b, se, z, p = report_term(m2, term)
    print(f"    {term:<15} β = {b:>10.5f}  SE = {se:>8.5f}  z = {z:>7.2f}  p = {p:>8.5f}   {stars(p)}")


# ============================================================================
# 5. PER-NETWORK FOLLOW-UP  (Bonferroni α = 0.00625 across 8 networks)
# ============================================================================
print("\n" + "=" * 78)
print("PER-NETWORK FOLLOW-UP — cross-sectional Btwn_cAge effect per network")
print("=" * 78)
cross_wide = df[df["cohort"].isin(["YA", "W1"])].copy()
cross_wide["mean_FD_c"] = cross_wide["mean_FD"] - cross_wide["mean_FD"].mean()
rows = []
hdr = f"  {'Net':<6} {'β(Btwn_cAge)':>14} {'SE':>10} {'z':>7} {'p':>9}    flag"
print(hdr); print("  " + "-"*60)
for net in NETWORKS:
    col = f"{net}_within"
    if col not in cross_wide or cross_wide[col].isna().all(): continue
    d = cross_wide.dropna(subset=[col, "between_cAge", "mean_FD_c"]).copy()
    m = smf.mixedlm(f"{col} ~ between_cAge + mean_FD_c",
                    data=d, groups=d["subject_id"]).fit(method="bfgs", reml=False)
    b, se, z, p = report_term(m, "between_cAge")
    rows.append({"model":"cross", "network":net, "n":int(m.nobs),
                 "beta":b, "se":se, "z":z, "p":p, "bonf_pass":p<ALPHA_BONF})
    print(f"  {net:<6} {b:>14.5f} {se:>10.5f} {z:>7.2f} {p:>9.5f}    {stars(p, bonf=True)}")


print("\n" + "=" * 78)
print("PER-NETWORK FOLLOW-UP — longitudinal With_cAge effect per network")
print("=" * 78)
long_wide = df[df["cohort"].isin(["W1", "W2"])].copy()
long_wide["mean_FD_c"] = long_wide["mean_FD"] - long_wide["mean_FD"].mean()
print(f"  {'Net':<6} {'β(With_cAge)':>14} {'SE':>10} {'z':>7} {'p':>9}    flag")
print("  " + "-"*60)
for net in NETWORKS:
    col = f"{net}_within"
    if col not in long_wide or long_wide[col].isna().all(): continue
    d = long_wide.dropna(subset=[col, "between_cAge", "within_cAge", "mean_FD_c"]).copy()
    m = None
    for opt in ("bfgs","lbfgs","powell","cg"):
        try:
            mm = smf.mixedlm(f"{col} ~ between_cAge + within_cAge + mean_FD_c",
                              data=d, groups=d["SubNum"]).fit(method=opt, reml=False)
            if np.isfinite(mm.aic): m = mm; break
        except Exception: continue
    if m is None: print(f"  {net:<6}  (no convergence)"); continue
    b, se, z, p = report_term(m, "within_cAge")
    rows.append({"model":"long", "network":net, "n":int(m.nobs),
                 "beta":b, "se":se, "z":z, "p":p, "bonf_pass":p<ALPHA_BONF})
    print(f"  {net:<6} {b:>14.5f} {se:>10.5f} {z:>7.2f} {p:>9.5f}    {stars(p, bonf=True)}")

# ============================================================================
# 6. Save tabulated results
# ============================================================================
pd.DataFrame(rows).to_csv(OUT_DIR/"pernetwork_followup.csv", index=False)
print(f"\nFollow-up results saved -> {OUT_DIR/'pernetwork_followup.csv'}")
print(f"Bonferroni α = {ALPHA_BONF:.5f}")
