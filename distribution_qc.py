#!/usr/bin/env python3
"""
distribution_qc.py
==================
Replicates the original manuscript's "Distribution-based signal quality control"
(Methods ¶46): for each subject compute mean BOLD change, SD BOLD change,
mean FD, SD FD; flag any subject whose value on ANY of the four metrics is
> 3 SD from the GROUP mean. Flagged subjects are then reviewed for exclusion.

Group = cohort (YA / W1 / W2) computed separately, because the cohorts have
very different motion and pooling would simply flag OA wholesale. The paper
reports QC exclusions per group, consistent with this choice.

    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/distribution_qc.py

INPUT  : motion_audit/cohort_motion_audit.csv  (must contain mean_bold, std_bold,
         mean_fd, std_fd — re-run cohort_motion_audit.py first if missing)
OUTPUTS: motion_audit/distribution_qc_flagged.csv
         motion_audit/distribution_qc.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH    = Path("/home/violetz/CONN_revision/motion_audit/cohort_motion_audit.csv")
OUT_DIR     = Path("/home/violetz/CONN_revision/motion_audit")
FLAG_PATH   = OUT_DIR / "distribution_qc_flagged.csv"
PLOT_PATH   = OUT_DIR / "distribution_qc.png"

METRICS = {
    "mean_bold": "Mean BOLD signal change",
    "std_bold":  "SD of BOLD signal change",
    "mean_fd":   "Mean FD (mm)",
    "std_fd":    "SD of FD (mm)",
}
COHORTS = ["YA", "W1", "W2", "MCI"]
COLORS  = {"YA": "#4C72B0", "W1": "#DD8452", "W2": "#55A868", "MCI": "#C44E52"}
SD_CUT  = 3.0


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[df["mean_fd"].notna()].copy()
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns {missing}. Re-run cohort_motion_audit.py.")

    # per-cohort z-scores for each metric
    for m in METRICS:
        df[f"z_{m}"] = (
            df.groupby("cohort")[m]
              .transform(lambda s: (s - s.mean()) / s.std(ddof=1))
        )

    z_cols = [f"z_{m}" for m in METRICS]
    df["max_abs_z"] = df[z_cols].abs().max(axis=1)
    df["flagged"] = df["max_abs_z"] > SD_CUT
    df["flag_metrics"] = df.apply(
        lambda r: ";".join(m for m in METRICS if abs(r[f"z_{m}"]) > SD_CUT), axis=1
    )

    flagged = df[df["flagged"]].sort_values(["cohort", "max_abs_z"], ascending=[True, False])

    keep = ["subject_id", "cohort", *METRICS, *z_cols, "flag_metrics", "max_abs_z"]
    flagged[keep].to_csv(FLAG_PATH, index=False)

    # ---- print ----
    print(f"\nDistribution-based QC (¶46 replication) — >|{SD_CUT:.0f} SD| from cohort mean\n")
    for c in COHORTS:
        cf = flagged[flagged["cohort"] == c]
        n = (df["cohort"] == c).sum()
        print(f"--- {c} (n={n}) : {len(cf)} flagged for review ---")
        for _, r in cf.iterrows():
            zdetail = ", ".join(
                f"{m}={r[m]:.3f} (z={r[f'z_{m}']:+.2f})"
                for m in METRICS if abs(r[f"z_{m}"]) > SD_CUT
            )
            print(f"  {r['subject_id']:<10} | {zdetail}")
        if len(cf) == 0:
            print("  (none)")
    print(f"\nTotal flagged for review: {len(flagged)} / {len(df)}")

    # ---- figure: 2x2, one panel per metric, per-cohort jitter + ±3SD band ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rng = np.random.default_rng(0)
    for ax, (m, label) in zip(axes.flat, METRICS.items()):
        for i, c in enumerate(COHORTS):
            sub = df[df["cohort"] == c]
            mu, sd = sub[m].mean(), sub[m].std(ddof=1)
            x = i + rng.uniform(-0.12, 0.12, len(sub))
            normal = sub[abs(sub[f"z_{m}"]) <= SD_CUT]
            outl   = sub[abs(sub[f"z_{m}"]) >  SD_CUT]
            ax.scatter(x[normal.index.map(sub.index.get_loc)], normal[m],
                       s=14, color=COLORS[c], alpha=0.6, edgecolor="none")
            if len(outl):
                ox = i + rng.uniform(-0.12, 0.12, len(outl))
                ax.scatter(ox, outl[m], s=55, color="red", edgecolor="black",
                           linewidth=0.6, zorder=4)
                for _, rr in outl.iterrows():
                    ax.annotate(rr["subject_id"], (i, rr[m]), fontsize=7,
                                xytext=(6, 0), textcoords="offset points", va="center")
            ax.hlines(mu, i - 0.25, i + 0.25, color="black", lw=1.5)
            ax.fill_between([i - 0.25, i + 0.25], mu - SD_CUT * sd, mu + SD_CUT * sd,
                            color=COLORS[c], alpha=0.10, zorder=0)
        ax.set_xticks(range(len(COHORTS)))
        ax.set_xticklabels(COHORTS)
        ax.set_title(label)
        ax.set_ylabel(m)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Distribution-based signal QC — per-cohort, points outside ±3 SD flagged (red)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOT_PATH, dpi=150)

    print(f"\nFlagged list saved → {FLAG_PATH}")
    print(f"Figure saved → {PLOT_PATH}")


if __name__ == "__main__":
    main()
