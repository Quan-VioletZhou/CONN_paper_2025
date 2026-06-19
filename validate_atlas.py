#!/usr/bin/env python3
"""
validate_atlas.py
=================
Validation of the merged Schaefer-300 + Buckner-7 atlas.

Checks:
  1. ROI counts per network match the published Schaefer-300 7Networks reference
  2. Left/right hemisphere balance is correct
  3. Per-ROI voxel-count distribution is sensible (no near-empty, no giant)
  4. Each ROI has a single connected location (no fragmented "ROIs" floating)
  5. ROI centers of mass land in the expected anatomical region
  6. Per-network spatial figure: each network rendered separately for eyeball QC

    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/validate_atlas.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import plotting

ATLAS  = Path("/home/violetz/CONN_revision/atlas/Schaefer300_Buckner7_2mm.nii.gz")
LABELS = Path("/home/violetz/CONN_revision/atlas/Schaefer300_Buckner7_labels.csv")
OUT_FIG = Path("/home/violetz/CONN_revision/atlas/atlas_validation.png")
OUT_COM = Path("/home/violetz/CONN_revision/atlas/atlas_roi_centers.csv")

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                 "Limbic", "Cont", "Default", "Cerebellar"]

# Published Schaefer-300 7Networks reference counts (from Schaefer 2018 supplementary)
EXPECTED_SCHAEFER = {
    "Vis": 47, "SomMot": 57, "DorsAttn": 34, "SalVentAttn": 34,
    "Limbic": 20, "Cont": 40, "Default": 68,
}


def main() -> None:
    labels = pd.read_csv(LABELS)
    img = nib.load(ATLAS)
    data = np.asarray(img.dataobj).astype(np.int32)
    affine = img.affine

    # ---- 1. Network counts vs reference ----
    print("=" * 64)
    print("1) ROI counts per network vs published Schaefer-300 7Networks")
    print("=" * 64)
    counts = labels["network"].value_counts().to_dict()
    print(f"   {'network':<13} {'observed':>9} {'expected':>9} {'match':>6}")
    all_match = True
    for n in NETWORK_ORDER:
        obs = counts.get(n, 0)
        exp = EXPECTED_SCHAEFER.get(n, 7 if n == "Cerebellar" else None)
        match = "OK" if exp is None or obs == exp else "X"
        if match == "X": all_match = False
        print(f"   {n:<13} {obs:>9} {exp if exp is not None else '-':>9} {match:>6}")
    print(f"   total ROIs: {len(labels)} (expect 307)")
    print(f"   verdict: {'ALL match' if all_match else 'MISMATCH'}")

    # ---- 2. Hemisphere balance (Schaefer ROIs only; Buckner is bilateral) ----
    print("\n" + "=" * 64)
    print("2) Hemisphere balance (Schaefer cortical ROIs)")
    print("=" * 64)
    sch = labels[labels["atlas_source"] == "Schaefer-300"]
    print(f"   LH: {(sch['hemisphere']=='LH').sum()}   "
          f"RH: {(sch['hemisphere']=='RH').sum()}   "
          f"(expect 150 / 150)")

    # ---- 3. Per-ROI voxel counts ----
    print("\n" + "=" * 64)
    print("3) Per-ROI voxel-count distribution (2 mm^3 each)")
    print("=" * 64)
    vcounts = np.array([(data == i).sum() for i in range(1, 308)])
    print(f"   min   = {vcounts.min():>5} voxels  ({vcounts.min()*8} mm^3)")
    print(f"   p05   = {int(np.percentile(vcounts,5)):>5}")
    print(f"   median= {int(np.median(vcounts)):>5}")
    print(f"   p95   = {int(np.percentile(vcounts,95)):>5}")
    print(f"   max   = {vcounts.max():>5} voxels  ({vcounts.max()*8} mm^3)")
    # split cortical vs cerebellar
    cort, cb = vcounts[:300], vcounts[300:]
    print(f"   cortical (1-300): min={cort.min()} median={int(np.median(cort))} max={cort.max()}")
    print(f"   cerebellar (301-307): {cb.tolist()}")

    # ---- 4. Connectedness ----
    print("\n" + "=" * 64)
    print("4) Connectedness: each ROI = 1 connected component?")
    print("=" * 64)
    fragmented = []
    for i in range(1, 308):
        mask = (data == i)
        if mask.sum() == 0:
            continue
        _, n = ndimage.label(mask)
        if n > 1:
            fragmented.append((i, n))
    print(f"   fragmented ROIs (>1 connected component): {len(fragmented)} / 307")
    if fragmented[:5]:
        for roi, n in fragmented[:5]:
            print(f"     ROI {roi}: {n} components")
    if len(fragmented) > 5:
        print(f"     ... and {len(fragmented)-5} more")

    # ---- 5. Centers of mass ----
    print("\n" + "=" * 64)
    print("5) Centers of mass (MNI mm) for first/last & all cerebellar")
    print("=" * 64)
    coms = []
    for i in range(1, 308):
        mask = data == i
        if mask.sum() == 0:
            coms.append((np.nan, np.nan, np.nan))
            continue
        ijk = np.array(ndimage.center_of_mass(mask))           # voxel coords
        xyz = nib.affines.apply_affine(affine, ijk)
        coms.append(tuple(xyz))
    coms = np.array(coms)
    com_df = labels.copy()
    com_df["mni_x"], com_df["mni_y"], com_df["mni_z"] = coms[:,0], coms[:,1], coms[:,2]
    com_df.to_csv(OUT_COM, index=False)
    for i in [1, 2, 150, 299, 300, 301, 304, 307]:
        r = com_df.iloc[i-1]
        print(f"   ROI {i:>3} ({r['network']:<11} {r['hemisphere']:<9}) "
              f"COM = ({r['mni_x']:+6.1f}, {r['mni_y']:+6.1f}, {r['mni_z']:+6.1f})")

    # ---- 6. Per-network figure ----
    print("\n" + "=" * 64)
    print("6) Per-network spatial visualization (figure)")
    print("=" * 64)
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for ax, net in zip(axes.flat, NETWORK_ORDER):
        ids = labels.loc[labels["network"] == net, "roi_id"].to_numpy()
        mask = np.isin(data, ids).astype(np.int16)
        net_img = nib.Nifti1Image(mask, affine, img.header)
        coords = (-2, -60, -30) if net == "Cerebellar" else None
        plotting.plot_roi(net_img, axes=ax, draw_cross=False, display_mode="ortho",
                          cmap="autumn", cut_coords=coords,
                          title=f"{net}  (n={len(ids)} ROIs)")
    fig.suptitle("Per-network spatial QC -- each panel = one of 8 networks", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"   figure saved -> {OUT_FIG}")
    print(f"   centers CSV  -> {OUT_COM}")


if __name__ == "__main__":
    main()
