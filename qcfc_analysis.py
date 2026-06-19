#!/usr/bin/env python3
"""
qcfc_analysis.py
================
Quality-control–functional-connectivity (QC-FC) analyses, following Ciric et al.
(2018) and Power et al. (2012), to verify that residual head motion does not
drive functional connectivity estimates after denoising.

Computes:
  1. Network-level QC-FC : Pearson r between each participant's mean FD and each
     network-level measure (within-network FC, between-network FC, segregation).
  2. Edge-level QC-FC    : Pearson r between mean FD and each of the 46,971
     region-to-region edges, across participants. Reports the distribution
     (median, % significant uncorrected and FDR) and QC-FC distance dependence
     (Pearson r between edgewise QC-FC and inter-regional Euclidean distance).

    module load python3.11-anaconda/2024.02
    python3 qcfc_analysis.py

OUTPUTS:
  qcfc_edge_r.npy          edgewise QC-FC correlations
  qcfc_edge_dist.npy       edgewise inter-regional distances
  network_qcfc.csv         network-level QC-FC table
  FigS_edge_QCFC.png       distribution + distance-dependence figure
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path("/home/violetz/CONN_revision")
FCDIR     = ROOT / "fc_nilearn"
MASTER    = ROOT / "CONNBx_Mastersheet_2026_6.5_final_outlier_removed.csv"
AUDIT     = ROOT / "motion_audit" / "cohort_motion_audit.csv"
CENTERS   = ROOT / "atlas" / "atlas_roi_centers.csv"
LABELS    = ROOT / "atlas" / "Schaefer300_Buckner7_labels.csv"
OUTDIR    = ROOT / "paper_revision"

NETS = ["DMN", "SMN", "VIS", "SAL", "DAN", "FP", "LIM", "CRB"]
NEW2SHORT = {"Vis": "VIS", "SomMot": "SMN", "DorsAttn": "DAN", "SalVentAttn": "SAL",
             "Limbic": "LIM", "Cont": "FP", "Default": "DMN", "Cerebellar": "CRB"}


def network_measures(fcz, net_idx):
    within = {n: np.nanmean(fcz[np.ix_(net_idx[n], net_idx[n])][np.triu_indices(len(net_idx[n]), k=1)])
              for n in NETS}
    pair = {}
    for a, b in combinations(NETS, 2):
        v = float(np.nanmean(fcz[np.ix_(net_idx[a], net_idx[b])]))
        pair[(a, b)] = v; pair[(b, a)] = v
    between = {n: np.mean([pair[(n, m)] for m in NETS if m != n]) for n in NETS}
    seg = {n: (within[n] - between[n]) / within[n] for n in NETS}
    row = {}
    for n in NETS:
        row[f"{n}_within"] = within[n]; row[f"{n}_between"] = between[n]; row[f"{n}_seg"] = seg[n]
    return row


def main():
    ms = pd.read_csv(MASTER)
    audit = pd.read_csv(AUDIT)
    fd_map = dict(zip(audit.subject_id, audit.mean_fd))

    # analytic sample: healthy controls, motion-clean, outlier-removed
    d = ms[(ms["cohort"].isin(["YA", "W1", "W2"])) &
           (ms["Subgroup"] == 1) & (ms["exclude"] == 0)].copy()
    d["mean_fd"] = d["Subject"].map(fd_map)
    d = d.dropna(subset=["mean_fd"])
    print(f"QC-FC sample: {len(d)} participants")

    # --- 1. network-level QC-FC (FD vs within/between/seg) ---
    rows = []
    for fam, suff in [("within", "_within"), ("between", "_between"), ("segregation", "_seg")]:
        for n in NETS:
            col = f"{n}{suff}"
            if col in d.columns:
                dd = d.dropna(subset=[col])
                r, p = pearsonr(dd["mean_fd"], dd[col])
                rows.append({"measure_family": fam, "network": n, "r_FD": r, "p": p})
    net_tab = pd.DataFrame(rows)
    net_tab.to_csv(OUTDIR / "network_qcfc.csv", index=False)
    print("Network-level QC-FC saved -> network_qcfc.csv")
    for fam in ["within", "between", "segregation"]:
        sub = net_tab[net_tab.measure_family == fam]
        print(f"  {fam:<12} |mean r| = {sub.r_FD.abs().mean():.3f}, "
              f"n sig(p<.05) = {(sub.p < .05).sum()}/8")

    # --- 2. edge-level QC-FC + distance dependence ---
    centers = pd.read_csv(CENTERS)
    coords = centers.set_index("roi_id").loc[range(1, 308), ["mni_x", "mni_y", "mni_z"]].values
    lab = pd.read_csv(LABELS)
    net_idx = {s: (lab.loc[lab.network == l, "roi_id"].to_numpy() - 1)
               for l, s in NEW2SHORT.items()}

    subs, fds, mats = [], [], []
    for _, r in d.iterrows():
        f = FCDIR / f"{r['Subject']}_fcz.npy"
        if f.exists():
            subs.append(r["Subject"]); fds.append(r["mean_fd"]); mats.append(np.load(f))
    fd = np.array(fds)

    iu = np.triu_indices(307, k=1)
    edge_fc = np.vstack([m[iu] for m in mats])
    edge_dist = np.sqrt(((coords[iu[0]] - coords[iu[1]]) ** 2).sum(1))
    good = ~np.isnan(edge_fc).any(0)

    qcfc = np.full(edge_fc.shape[1], np.nan); qp = np.full(edge_fc.shape[1], np.nan)
    for e in np.where(good)[0]:
        qcfc[e], qp[e] = pearsonr(edge_fc[:, e], fd)
    qv, pv, dv = qcfc[good], qp[good], edge_dist[good]
    rej, _, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
    rdist, _ = pearsonr(qv, dv)

    print("\nEdge-level QC-FC:")
    print(f"  median QC-FC r        : {np.median(qv):+.4f}")
    print(f"  absolute median       : {np.median(np.abs(qv)):.4f}")
    print(f"  % edges p<.05 (uncorr): {100*np.mean(pv < .05):.1f}%")
    print(f"  % edges sig (FDR)     : {100*np.mean(rej):.1f}%")
    print(f"  distance dependence r : {rdist:+.3f}")

    np.save(OUTDIR / "qcfc_edge_r.npy", qcfc)
    np.save(OUTDIR / "qcfc_edge_dist.npy", edge_dist)

    # --- figure: distribution + distance dependence ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(qv, bins=80, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(np.median(qv), color="#cd3e4e", lw=1.6, ls="--",
               label=f"median = {np.median(qv):+.3f}")
    ax.set_xlabel("QC-FC correlation (r)"); ax.set_ylabel("Number of edges")
    ax.set_title("(a) Distribution of edge-level QC-FC correlations"); ax.legend()
    ax = axes[1]
    idx = np.random.default_rng(0).choice(len(qv), size=min(8000, len(qv)), replace=False)
    ax.scatter(dv[idx], qv[idx], s=3, alpha=0.15, color="#4C72B0")
    z = np.polyfit(dv, qv, 1); xs = np.linspace(dv.min(), dv.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color="#cd3e4e", lw=2, label=f"r = {rdist:+.3f}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Inter-regional Euclidean distance (mm)")
    ax.set_ylabel("QC-FC correlation (r)")
    ax.set_title("(b) QC-FC distance dependence"); ax.legend()
    fig.suptitle(f"Edge-level QC-FC benchmarks (n = {len(subs)}, {good.sum():,} edges)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / "FigS_edge_QCFC.png", dpi=160, bbox_inches="tight")
    print(f"\nFigure saved -> FigS_edge_QCFC.png")


if __name__ == "__main__":
    main()
