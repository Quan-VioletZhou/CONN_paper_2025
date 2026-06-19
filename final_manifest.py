#!/usr/bin/env python3
"""
final_manifest.py
=================
Locks the motion-stage subject decision using the agreed objective rule
(NO manual review):

    EXCLUDE if  (< 50% high-quality volumes)
             OR (> 3 SD from cohort mean on any of
                 {mean BOLD, SD BOLD, mean FD, SD FD}  -- ¶46 distribution QC)

Produces:
  motion_audit/excluded_subjects.csv  : every excluded subject + reason
  motion_audit/included_manifest.csv  : the kept subjects (atlas-extraction input)
  motion_audit/included_manifest.txt  : plain SID list, one per line per cohort

    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/final_manifest.py
"""

from pathlib import Path
import pandas as pd

OUT_DIR   = Path("/home/violetz/CONN_revision/motion_audit")
AUDIT     = OUT_DIR / "cohort_motion_audit.csv"
FLAGGED   = OUT_DIR / "distribution_qc_flagged.csv"
EXCL_CSV  = OUT_DIR / "excluded_subjects.csv"
KEEP_CSV  = OUT_DIR / "included_manifest.csv"
KEEP_TXT  = OUT_DIR / "included_manifest.txt"

HQ_THRESH = 50
COHORTS   = ["YA", "W1", "W2", "MCI"]


def main() -> None:
    df = pd.read_csv(AUDIT)
    df = df[df["mean_fd"].notna()].copy()
    fl = pd.read_csv(FLAGGED)[["subject_id", "flag_metrics"]]
    df = df.merge(fl, on="subject_id", how="left")

    df["fail_thresh"] = df["pct_high_quality"] < HQ_THRESH
    df["qc_dist"]     = df["subject_id"].isin(set(fl["subject_id"]))
    df["exclude"]     = df["fail_thresh"] | df["qc_dist"]

    def reason(r):
        if r["fail_thresh"] and r["qc_dist"]:
            return "thresh+qc_dist"
        if r["fail_thresh"]:
            return "thresh (<50% HQ)"
        if r["qc_dist"]:
            return "qc_dist (¶46 >3SD)"
        return ""
    df["exclude_reason"] = df.apply(reason, axis=1)

    excl = df[df["exclude"]].sort_values(
        ["cohort", "exclude_reason", "pct_high_quality"]
    )
    keep = df[~df["exclude"]].sort_values(["cohort", "subject_id"])

    excl_cols = ["subject_id", "cohort", "scan_session", "pct_high_quality",
                 "mean_fd", "max_fd", "max_abs_bold", "exclude_reason", "flag_metrics"]
    excl[excl_cols].to_csv(EXCL_CSV, index=False)
    keep[["subject_id", "cohort", "scan_session",
          "pct_high_quality", "mean_fd"]].to_csv(KEEP_CSV, index=False)
    with KEEP_TXT.open("w") as f:
        for c in COHORTS:
            ids = keep.loc[keep["cohort"] == c, "subject_id"].tolist()
            f.write(f"# {c} (n={len(ids)})\n")
            f.write("\n".join(ids) + "\n\n")

    # ---- printed excluded table ----
    print(f"\nEXCLUDED SUBJECTS  (rule: <{HQ_THRESH}% HQ  OR  ¶46 >3SD distribution QC)\n")
    for c in COHORTS:
        ce = excl[excl["cohort"] == c]
        nA = (df["cohort"] == c).sum()
        nK = (keep["cohort"] == c).sum()
        print(f"--- {c}: {len(ce)} excluded  (audited {nA} -> kept {nK}) ---")
        print(f"  {'subject':<10} {'%HQ':>6} {'meanFD':>7} {'maxFD':>6} {'|BOLD|max':>9}  reason")
        for _, r in ce.iterrows():
            extra = f"  [{r['flag_metrics']}]" if isinstance(r["flag_metrics"], str) and r["flag_metrics"] else ""
            print(f"  {r['subject_id']:<10} {r['pct_high_quality']:>6.1f} "
                  f"{r['mean_fd']:>7.3f} {r['max_fd']:>6.2f} {r['max_abs_bold']:>9.2f}  "
                  f"{r['exclude_reason']}{extra}")
        print()

    print("=" * 60)
    print(f"{'':<10}{'audited':>9}{'excluded':>10}{'kept':>7}")
    for c in COHORTS:
        nA = int((df['cohort'] == c).sum())
        nE = int((excl['cohort'] == c).sum())
        print(f"{c:<10}{nA:>9}{nE:>10}{nA - nE:>7}")
    print(f"{'TOTAL':<10}{len(df):>9}{len(excl):>10}{len(keep):>7}")
    print("=" * 60)
    print(f"\nExcluded list  -> {EXCL_CSV}")
    print(f"Kept manifest  -> {KEEP_CSV}")
    print(f"Kept SID list  -> {KEEP_TXT}")


if __name__ == "__main__":
    main()
