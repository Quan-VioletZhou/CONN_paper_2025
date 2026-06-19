#!/usr/bin/env python3
"""
cohort_motion_audit.py
======================
Cohort-level motion audit for the IMAG-26-0103 revision (reviewer asks R1a / R3.2).

Iterates over all subjects in the three cohorts (YA, W1, W2), reads Echo's
per-subject ART outputs, and produces:
  - cohort_motion_audit.csv : one row per subject with all motion stats
  - printed aggregate summary (per-cohort means, exclusion counts at each threshold)

USAGE
-----
    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/cohort_motion_audit.py

INPUTS
------
  Subject lists derived from folder names in /nfs/turbo/.../mind/subjects/
  (prefix mindy=YA, mindo=W1, mindb=W2). Motion data also read from there:
      /nfs/turbo/.../mind/subjects/<SID>/{placebo|placebodti}/func/connectivity/conn_proj/Func/
          art_regression_timeseries_aurun_01.mat   # cols: [BOLD (SD), FD (mm)]
          art_regression_outliers_aurun_01.mat     # one col per flagged volume

OUTPUTS
-------
  /home/violetz/CONN_revision/motion_audit/cohort_motion_audit.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

# =============================================================================
# Paths and constants
# =============================================================================
SUBJECTS_ROOT  = Path("/nfs/turbo/lsa-tpolk/tpolk-mistorage/mind/subjects")
OUT_DIR        = Path("/home/violetz/CONN_revision/motion_audit")
CSV_PATH       = OUT_DIR / "cohort_motion_audit.csv"

COHORTS = {
    "YA":  "mindy",
    "W1":  "mindo",
    "W2":  "mindb",
    "MCI": "mindm",
}

FD_INCL_THRESH    = 0.2
FD_SCRUB_THRESH   = 0.45
BOLD_SCRUB_THRESH = 5.0
INCLUSION_PCTS    = (20, 30, 40, 50)


def list_subjects(sid_prefix: str) -> list[str]:
    """Subjects in a cohort = all folders under /subjects/ matching the prefix."""
    pattern = re.compile(rf"^{sid_prefix}\d+$")
    return sorted(d.name for d in SUBJECTS_ROOT.iterdir() if pattern.match(d.name))


def find_func_dir(sid: str) -> Path | None:
    for sub in ("placebo", "placebodti"):
        p = SUBJECTS_ROOT / sid / sub / "func" / "connectivity" / "conn_proj" / "Func"
        if (p / "art_regression_timeseries_aurun_01.mat").exists():
            return p
    return None


def audit_subject(sid: str, cohort: str) -> dict:
    row = {"subject_id": sid, "cohort": cohort}
    func_dir = find_func_dir(sid)
    if func_dir is None:
        row["error"] = "no Echo output found"
        return row
    row["scan_session"] = func_dir.parts[-5]  # placebo or placebodti

    ts_file  = func_dir / "art_regression_timeseries_aurun_01.mat"
    out_file = func_dir / "art_regression_outliers_aurun_01.mat"
    try:
        R_ts  = sio.loadmat(str(ts_file))["R"]
        R_out = sio.loadmat(str(out_file))["R"]
    except Exception as e:
        row["error"] = f"load failed: {e}"
        return row

    bold = R_ts[:, 0]
    fd   = R_ts[:, 1]
    n_vols = R_ts.shape[0]
    n_art_outliers = R_out.shape[1] if R_out.ndim == 2 else 0

    high_quality_mask = (fd < FD_INCL_THRESH) & (np.abs(bold) < BOLD_SCRUB_THRESH)
    pct_high_quality = 100.0 * high_quality_mask.sum() / n_vols

    row.update({
        "n_vols":             int(n_vols),
        "mean_fd":            float(fd.mean()),
        "max_fd":             float(fd.max()),
        "std_fd":             float(fd.std()),
        "mean_bold":          float(bold.mean()),
        "std_bold":           float(bold.std()),
        "max_abs_bold":       float(np.abs(bold).max()),
        "n_fd_above_0.2":     int((fd > FD_INCL_THRESH).sum()),
        "n_fd_above_0.45":    int((fd > FD_SCRUB_THRESH).sum()),
        "n_bold_above_5sd":   int((np.abs(bold) > BOLD_SCRUB_THRESH).sum()),
        "n_art_outliers":     int(n_art_outliers),
        "pct_high_quality":   float(pct_high_quality),
        "error":              "",
    })
    for thresh in INCLUSION_PCTS:
        row[f"pass_{thresh}pct"] = bool(pct_high_quality >= thresh)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "subject_id", "cohort", "scan_session",
        "n_vols", "mean_fd", "max_fd", "std_fd",
        "mean_bold", "std_bold", "max_abs_bold",
        "n_fd_above_0.2", "n_fd_above_0.45", "n_bold_above_5sd",
        "n_art_outliers", "pct_high_quality",
        *(f"pass_{t}pct" for t in INCLUSION_PCTS),
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                elif isinstance(v, bool):
                    vals.append("1" if v else "0")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)
    for cohort in COHORTS:
        cr = [r for r in rows if r["cohort"] == cohort]
        good = [r for r in cr if not r.get("error")]
        missing = [r for r in cr if r.get("error")]
        print(f"\n--- {cohort} ({len(cr)} subjects total) ---")
        print(f"  successful: {len(good)} | missing/failed: {len(missing)}")
        if missing:
            print(f"  missing SIDs: {[r['subject_id'] for r in missing]}")
        if not good:
            continue
        mfd  = np.array([r["mean_fd"] for r in good])
        mxfd = np.array([r["max_fd"]  for r in good])
        phq  = np.array([r["pct_high_quality"] for r in good])
        nart = np.array([r["n_art_outliers"] for r in good])
        print(f"  mean FD (mm):       mean={mfd.mean():.3f}  SD={mfd.std():.3f}  range=[{mfd.min():.3f}, {mfd.max():.3f}]")
        print(f"  max FD (mm):        mean={mxfd.mean():.3f}  SD={mxfd.std():.3f}  range=[{mxfd.min():.3f}, {mxfd.max():.3f}]")
        print(f"  % high-quality:     mean={phq.mean():.1f}  SD={phq.std():.1f}  range=[{phq.min():.1f}, {phq.max():.1f}]")
        print(f"  # ART outliers:     mean={nart.mean():.1f}  SD={nart.std():.1f}  range=[{nart.min()}, {nart.max()}]")
        print(f"  inclusion (% subjects passing each threshold):")
        for thresh in INCLUSION_PCTS:
            n_pass = sum(r[f"pass_{thresh}pct"] for r in good)
            print(f"      ≥{thresh}% high-quality: {n_pass}/{len(good)} pass  ({n_pass/len(good)*100:.0f}%)")


def main() -> None:
    print(f"Cohort motion audit — reading from {SUBJECTS_ROOT}")
    print(f"Cohort = all folders matching mindy/mindo/mindb prefixes\n")

    all_rows = []
    for cohort, prefix in COHORTS.items():
        sids = list_subjects(prefix)
        print(f"  {cohort} ({prefix}*): {len(sids)} subjects → auditing...")
        for sid in sids:
            all_rows.append(audit_subject(sid, cohort))

    write_csv(all_rows, CSV_PATH)
    print(f"\nCSV written → {CSV_PATH}")
    summarize(all_rows)


if __name__ == "__main__":
    main()
