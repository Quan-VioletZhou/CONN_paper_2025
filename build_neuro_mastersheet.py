#!/usr/bin/env python3
"""
build_neuro_mastersheet.py
==========================
Combined NEURO-ONLY master sheet for all subjects with a Fisher-z FC matrix
in fc_nilearn/.  Demographics/behavioral columns are intentionally left out —
the RA will merge those in later from their own source.

ROWS = every subject_id with fc_nilearn/<sid>_fcz.npy  (n = 292 expected:
       53 YA + 144 W1 + 51 W2 + 44 MCI).

COLUMNS:
  subject_id        e.g. mindy104, mindo182, mindb251, mindm310
  cohort            YA / W1 / W2 / MCI  (inferred from prefix mindy/mindo/mindb/mindm)
  exclude           0 = passes motion rule, 1 = excluded
  exclude_reason    one of {"", "thresh (<50% HQ)", "qc_dist (¶46 >3SD)", "thresh+qc_dist"}
  pct_high_quality  % of volumes with FD < 0.2 AND |BOLD| < 5 SD (from ART)
  mean_FD           subject-mean FD (mm)

  Per-network FC summaries — 8 networks (LAN dropped; Limbic kept as LIM):
    {DMN, SMN, VIS, SAL, DAN, FP, LIM, CRB}
       N_within     mean Fisher-z over ROI pairs *within* network N
       N_between    mean Fisher-z over all pairs (N, M ≠ N)
       N_seg        (N_within − N_between) / N_within
  Global summaries:
       global_within / global_between / global_seg

  Pairwise between-network FC — 6 pairs among {DMN, DAN, FP, SAL}:
       DMN_DAN, DMN_FP, DMN_SAL, DAN_FP, DAN_SAL, FP_SAL
       (each = mean Fisher-z over ROI pairs across the two listed networks)

USAGE
-----
    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/build_neuro_mastersheet.py

OUTPUT
------
    motion_audit/CONN_NeuroMastersheet_2026.csv
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT      = Path("/home/violetz/CONN_revision")
FCDIR     = ROOT / "fc_nilearn"
LABELS    = ROOT / "atlas" / "Schaefer300_Buckner7_labels.csv"
AUDIT     = ROOT / "motion_audit" / "cohort_motion_audit.csv"
EXCLUDED  = ROOT / "motion_audit" / "excluded_subjects.csv"
OUT       = ROOT / "motion_audit" / "CONN_NeuroMastersheet_2026.csv"

# new-atlas network name  -> short label used in output columns
NEW2SHORT = {
    "Vis":         "VIS",
    "SomMot":      "SMN",
    "DorsAttn":    "DAN",
    "SalVentAttn": "SAL",
    "Limbic":      "LIM",         # <-- renamed from "Limbic" to "LIM"
    "Cont":        "FP",
    "Default":     "DMN",
    "Cerebellar":  "CRB",
}
NET_ORDER  = ["DMN", "SMN", "VIS", "SAL", "DAN", "FP", "LIM", "CRB"]   # output order
NEW_ORDER  = [k for k in ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default","Cerebellar"]]
PAIR_NETS  = ["DMN", "DAN", "FP", "SAL"]                              # 6 pairwise: C(4,2)
COHORT_PFX = {"mindy": "YA", "mindo": "W1", "mindb": "W2", "mindm": "MCI"}


# ---- 1. Build network -> ROI-index map from the atlas label file ----------
lab = pd.read_csv(LABELS)
net_idx = {short: (lab.loc[lab["network"] == longname, "roi_id"].to_numpy() - 1)
           for longname, short in NEW2SHORT.items()}


# ---- 2. Read motion + exclusion tables -----------------------------------
audit = pd.read_csv(AUDIT)[["subject_id", "pct_high_quality", "mean_fd"]]
audit = audit.rename(columns={"mean_fd": "mean_FD"})

excl = pd.read_csv(EXCLUDED)[["subject_id", "exclude_reason"]]
excl["exclude"] = 1
excl_map      = dict(zip(excl["subject_id"], excl["exclude_reason"]))


# ---- 3. Loop over every fc_nilearn subject and compute FC summaries ------
def fc_summary(fcz: np.ndarray) -> dict:
    """Compute per-net within/between/seg, global, and 6 pairwise FCs."""
    out = {}

    # per-network within (upper-triangle mean over ROIs in that net)
    within = {}
    for net in NET_ORDER:
        idx = net_idx[net]
        if len(idx) > 1:
            block = fcz[np.ix_(idx, idx)]
            within[net] = float(np.nanmean(block[np.triu_indices(len(idx), k=1)]))
        else:
            within[net] = np.nan

    # all pairwise between-network FCs (store both orderings for easy lookup)
    pair_bw = {}
    for a, b in combinations(NET_ORDER, 2):
        v = float(np.nanmean(fcz[np.ix_(net_idx[a], net_idx[b])]))
        pair_bw[(a, b)] = v
        pair_bw[(b, a)] = v

    # per-network between = mean of its pairwise BTs with the other 7
    between = {n: float(np.mean([pair_bw[(n, m)] for m in NET_ORDER if m != n]))
               for n in NET_ORDER}

    # segregation
    seg = {n: (within[n] - between[n]) / within[n] if within[n] not in (0, np.nan) else np.nan
           for n in NET_ORDER}

    for n in NET_ORDER:
        out[f"{n}_within"]  = within[n]
        out[f"{n}_between"] = between[n]
        out[f"{n}_seg"]     = seg[n]

    # global = mean across the per-net summaries
    out["global_within"]  = float(np.mean([within[n]  for n in NET_ORDER]))
    out["global_between"] = float(np.mean([pair_bw[(a,b)] for a, b in combinations(NET_ORDER, 2)]))
    out["global_seg"]     = ((out["global_within"] - out["global_between"]) / out["global_within"]
                              if out["global_within"] else np.nan)

    # 6 pairwise between-network FCs among {DMN, DAN, FP, SAL}
    for a, b in combinations(PAIR_NETS, 2):
        out[f"{a}_{b}"] = pair_bw[(a, b)]

    return out


rows = []
for f in sorted(FCDIR.glob("*_fcz.npy")):
    sid    = f.stem.replace("_fcz", "")
    prefix = sid[:5]
    cohort = COHORT_PFX.get(prefix)
    if cohort is None:
        continue                                # unknown prefix → skip

    fcz = np.load(f)                            # (307, 307) Fisher z, diag = NaN
    row = {"subject_id": sid, "cohort": cohort}
    row.update(fc_summary(fcz))
    rows.append(row)

fc_df = pd.DataFrame(rows)
print(f"FC rows assembled : {len(fc_df)}")
print(f"  by cohort       : {fc_df['cohort'].value_counts().to_dict()}")


# ---- 4. Merge motion (mean_FD, pct_high_quality) and exclude flag --------
df = fc_df.merge(audit, on="subject_id", how="left")
df["exclude"]        = df["subject_id"].isin(excl_map).astype(int)
df["exclude_reason"] = df["subject_id"].map(excl_map).fillna("")


# ---- 5. Final column order -----------------------------------------------
id_cols      = ["subject_id", "cohort", "exclude", "exclude_reason",
                "pct_high_quality", "mean_FD"]
within_cols  = [f"{n}_within"  for n in NET_ORDER]
between_cols = [f"{n}_between" for n in NET_ORDER]
seg_cols     = [f"{n}_seg"     for n in NET_ORDER]
global_cols  = ["global_within", "global_between", "global_seg"]
pair_cols    = [f"{a}_{b}" for a, b in combinations(PAIR_NETS, 2)]

final_cols = id_cols + within_cols + between_cols + seg_cols + global_cols + pair_cols
out_df = df[final_cols].copy()

# sort: cohort first, then subject_id
out_df = out_df.sort_values(
    by=["cohort", "subject_id"],
    key=lambda c: c.map({"YA": 0, "W1": 1, "W2": 2, "MCI": 3}) if c.name == "cohort" else c
).reset_index(drop=True)

out_df.to_csv(OUT, index=False)
print(f"\nsaved -> {OUT}")
print(f"shape : {out_df.shape}    ({len(out_df.columns)} columns)")

# ---- 6. Sanity report -----------------------------------------------------
print("\nExclude flag by cohort:")
print(out_df.groupby("cohort")["exclude"].agg(["count", "sum"])
      .rename(columns={"count":"n_subjects", "sum":"n_excluded"}))

print("\nFirst 3 columns × first 5 rows:")
print(out_df.iloc[:5, :6].to_string(index=False))

print("\nColumn list:")
for c in out_df.columns:
    print(f"  {c}")
