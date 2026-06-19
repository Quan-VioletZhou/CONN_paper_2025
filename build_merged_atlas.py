#!/usr/bin/env python3
"""
build_merged_atlas.py
=====================
Build the merged Schaefer-300 + Buckner-7 atlas for the IMAG-26-0103 revision
(B.2: parcellation modification; R1(b), R3.1).

  - Schaefer-300 (Yeo-7 networks), 2 mm MNI152, fetched via nilearn
  - Buckner-7 cerebellar (Buckner et al. 2011), local FreeSurfer-conformed 1 mm,
    resampled to the Schaefer grid (nearest-neighbour)
  - Merged into a single labelled NIfTI: labels 1-300 cortical, 301-307 cerebellar
  - Cortical wins on any voxel overlap (cerebellar is sparse where cortex exists)
  - Network membership: Schaefer ROIs keep their Yeo-7 label; ALL 7 Buckner ROIs
    are collapsed into a single "Cerebellar" network (locked decision -- see
    parcellation_decision memory). Network total = 8.

    module load python3.11-anaconda/2024.02
    python3 /home/violetz/CONN_revision/build_merged_atlas.py

OUTPUTS:
  atlas/Schaefer300_Buckner7_2mm.nii.gz   labelled atlas, 307 ROIs
  atlas/Schaefer300_Buckner7_labels.csv   per-ROI network membership
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image

OUT_DIR = Path("/home/violetz/CONN_revision/atlas")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKNER_NII = Path(
    "/nfs/turbo/lsa-tpolk/tpolk-mistorage/Sensorimotor_Dedifferentiation_Aging/"
    "freesurfer/average/Buckner_JNeurophysiol11_MNI152/"
    "Buckner2011_7Networks_MNI152_FreeSurferConformed1mm_TightMask.nii.gz"
)
NILEARN_DATA = OUT_DIR / "nilearn_data"

OUT_ATLAS  = OUT_DIR / "Schaefer300_Buckner7_2mm.nii.gz"
OUT_LABELS = OUT_DIR / "Schaefer300_Buckner7_labels.csv"

BUCKNER_NETWORK_NAMES = {
    1: "Buckner_Visual",     2: "Buckner_SomMot",    3: "Buckner_DorsAttn",
    4: "Buckner_SalVentAttn",5: "Buckner_Limbic",    6: "Buckner_Cont",
    7: "Buckner_Default",
}


def parse_schaefer_label(s: str) -> dict:
    # e.g.  '7Networks_LH_Vis_1'  ->  hemi LH, network Vis
    s = s if isinstance(s, str) else s.decode()
    parts = s.split("_")
    return {"hemisphere": parts[1], "network_short": parts[2], "raw_label": s}


def main() -> None:
    print("Fetching Schaefer-300 (Yeo-7, 2 mm) via nilearn ...")
    sch = datasets.fetch_atlas_schaefer_2018(
        n_rois=300, yeo_networks=7, resolution_mm=2, data_dir=str(NILEARN_DATA)
    )
    sch_img = nib.load(sch.maps)
    sch_data = np.asarray(sch_img.dataobj).astype(np.int32)
    print(f"  Schaefer image: shape={sch_img.shape}, zooms={sch_img.header.get_zooms()}")
    print(f"  unique labels (first/last): {sch_data.min()} ... {sch_data.max()}, "
          f"n_unique={len(np.unique(sch_data))}")

    print(f"\nLoading Buckner 7Networks (1 mm) and resampling to Schaefer grid ...")
    buck_img = nib.load(BUCKNER_NII)
    buck_resamp = image.resample_to_img(
        buck_img, sch_img, interpolation="nearest", force_resample=True, copy_header=True
    )
    buck_data = np.asarray(buck_resamp.dataobj).astype(np.int32)
    if buck_data.ndim == 4:
        buck_data = buck_data[..., 0]
    print(f"  Buckner resampled: shape={buck_data.shape}, "
          f"unique labels: {np.unique(buck_data)}")

    print(f"\nMerging (cortical wins on overlap) ...")
    merged = sch_data.copy()
    cortical_mask = merged > 0
    buck_mask     = buck_data > 0
    overlap_n = int(np.sum(cortical_mask & buck_mask))
    add_mask  = buck_mask & ~cortical_mask
    merged[add_mask] = buck_data[add_mask] + 300
    print(f"  overlap voxels (Schaefer & Buckner): {overlap_n}  (cortical retained)")
    print(f"  cerebellar voxels added              : {int(add_mask.sum())}")
    print(f"  final unique labels                  : {np.unique(merged).size} "
          f"(expect 308 = 0 + 307 ROIs)")

    out_img = nib.Nifti1Image(merged.astype(np.int16), sch_img.affine, sch_img.header)
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, OUT_ATLAS)
    print(f"  saved merged atlas -> {OUT_ATLAS}")

    # --- labels table ---
    rows = []
    for i in range(1, 301):
        meta = parse_schaefer_label(sch.labels[i])   # labels[0] = 'Background'
        rows.append({
            "roi_id": i,
            "atlas_source": "Schaefer-300",
            "hemisphere": meta["hemisphere"],
            "network": meta["network_short"],
            "raw_label": meta["raw_label"],
        })
    for k in range(1, 8):
        rows.append({
            "roi_id": 300 + k,
            "atlas_source": "Buckner-7",
            "hemisphere": "bilateral",
            "network": "Cerebellar",                       # collapsed (locked decision)
            "raw_label": BUCKNER_NETWORK_NAMES[k],          # keep the Buckner sub-id label
        })
    labels = pd.DataFrame(rows)
    labels.to_csv(OUT_LABELS, index=False)
    print(f"  saved labels CSV   -> {OUT_LABELS}")

    # --- summary ---
    print("\nNetwork ROI counts (8 networks expected):")
    print(labels["network"].value_counts().to_string())

    # --- sanity check: per-ROI voxel counts ---
    print("\nPer-ROI voxel count distribution (all 307 ROIs):")
    counts = np.array([int((merged == i).sum()) for i in range(1, 308)])
    print(f"  min={counts.min()} | median={int(np.median(counts))} | "
          f"max={counts.max()} | empty ROIs (0 voxels) = {(counts == 0).sum()}")
    if (counts == 0).any():
        empty_ids = np.where(counts == 0)[0] + 1
        print(f"  WARNING -- empty ROI ids: {empty_ids.tolist()}")


if __name__ == "__main__":
    main()
