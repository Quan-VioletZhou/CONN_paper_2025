#!/usr/bin/env python3
"""
compute_network_fc.py
=====================
First-level network connectivity from a subject's 307x307 Fisher-z FC matrix:
  - within-network FC      (8 values)
  - between-network FC     (8x8 network matrix)
  - network segregation    (8 values; (within - mean-between) / within)
Also renders the ROI-level FC heatmap (ordered by network) and the 8x8 network matrix.

    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/compute_network_fc.py mindo100
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path("/home/violetz/CONN_revision")
LABELS = ROOT / "atlas" / "Schaefer300_Buckner7_labels.csv"
FCDIR = ROOT / "fc"
ORDER = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default","Cerebellar"]

sid = sys.argv[1] if len(sys.argv) > 1 else "mindo100"
fcz = np.load(FCDIR / f"{sid}_fcz.npy")              # 307x307, diag NaN
lab = pd.read_csv(LABELS)

# order ROIs by network (then roi_id)
lab["net_rank"] = lab["network"].map({n:i for i,n in enumerate(ORDER)})
lab = lab.sort_values(["net_rank","roi_id"]).reset_index(drop=True)
order_idx = (lab["roi_id"].to_numpy() - 1)           # 0-based positions into fcz
fcz_ord = fcz[np.ix_(order_idx, order_idx)]
net_of = lab["network"].to_numpy()

# ---- network-level summaries ----
within, between = {}, {n: {} for n in ORDER}
for a in ORDER:
    ia = np.where(net_of == a)[0]
    sub = fcz_ord[np.ix_(ia, ia)]
    within[a] = np.nanmean(sub[np.triu_indices(len(ia), k=1)])
    for b in ORDER:
        if b == a: continue
        ib = np.where(net_of == b)[0]
        between[a][b] = np.nanmean(fcz_ord[np.ix_(ia, ib)])

seg = {a: (within[a] - np.mean(list(between[a].values()))) / within[a] for a in ORDER}

print(f"\nFirst-level network connectivity — {sid}  (Fisher z)\n")
print(f"  {'network':<12}{'within':>9}{'mean-between':>14}{'segregation':>13}")
for a in ORDER:
    mb = np.mean(list(between[a].values()))
    print(f"  {a:<12}{within[a]:>9.3f}{mb:>14.3f}{seg[a]:>13.3f}")
gw = np.mean([within[a] for a in ORDER]); gb = np.mean([between[a][b] for a in ORDER for b in ORDER if b!=a])
print(f"\n  global within={gw:.3f}  global between={gb:.3f}  global segregation={(gw-gb)/gw:.3f}")

# ---- figure: ROI heatmap (ordered) + 8x8 network matrix ----
fig, (axR, axN) = plt.subplots(1, 2, figsize=(15, 6.5),
                               gridspec_kw={"width_ratios":[1.4,1]})
M = fcz_ord.copy(); np.fill_diagonal(M, 0.0)
im = axR.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
# network boundaries + labels
bounds, pos = [], []
start = 0
for n in ORDER:
    cnt = int((net_of == n).sum()); end = start + cnt
    axR.axhline(end-0.5, color="k", lw=0.6); axR.axvline(end-0.5, color="k", lw=0.6)
    pos.append((start+end)/2); start = end
axR.set_xticks(pos); axR.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=9)
axR.set_yticks(pos); axR.set_yticklabels(ORDER, fontsize=9)
axR.set_title(f"{sid}: ROI-level FC (307×307, ordered by network)")
fig.colorbar(im, ax=axR, fraction=0.046, label="Fisher z")

netmat = np.array([[within[a] if a==b else between[a][b] for b in ORDER] for a in ORDER])
im2 = axN.imshow(netmat, cmap="RdBu_r", vmin=-0.4, vmax=0.6)
for i in range(8):
    for j in range(8):
        axN.text(j, i, f"{netmat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                 color="black" if abs(netmat[i,j])<0.4 else "white")
axN.set_xticks(range(8)); axN.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=9)
axN.set_yticks(range(8)); axN.set_yticklabels(ORDER, fontsize=9)
axN.set_title(f"{sid}: 8×8 network FC (diagonal = within)")
fig.colorbar(im2, ax=axN, fraction=0.046, label="Fisher z")
fig.tight_layout()
out = FCDIR / f"{sid}_fc_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nfigure -> {out}")
