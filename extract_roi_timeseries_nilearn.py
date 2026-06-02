#!/usr/bin/env python3
"""
extract_roi_timeseries_nilearn.py
=================================
First-level extraction using only standard libraries (nilearn + sklearn + numpy).
Parallel companion to extract_roi_timeseries.py (the custom-implementation
version, kept intact for comparison).

Pipeline:
  1. nilearn.maskers.NiftiLabelsMasker       -> mean voxel value per parcel
  2. nilearn.connectome.ConnectivityMeasure  -> Pearson r
       cov_estimator = EmpiricalCovariance() -> raw sample correlation
       (the ONLY non-default setting; opts out of the default Ledoit-Wolf shrinkage)
  3. numpy.arctanh                           -> Fisher z transform

Methods one-liner (drop in your paper):
  "ROI timeseries were extracted with nilearn.maskers.NiftiLabelsMasker (Abraham
   et al. 2014; mean voxel value per parcel). Pairwise connectivity was computed
   with nilearn.connectome.ConnectivityMeasure (kind='correlation',
   cov_estimator=EmpiricalCovariance, i.e., raw sample Pearson correlation),
   Fisher z-transformed for downstream analyses."

Outputs go to fc_nilearn/ (NOT the original fc/), so the two sets coexist for
verification before we choose which to feed downstream.

    module load python3.11-anaconda/2024.02
    python3 extract_roi_timeseries_nilearn.py                       # all 248 from manifest
    python3 extract_roi_timeseries_nilearn.py mindo100              # one subject
    python3 extract_roi_timeseries_nilearn.py --workers 16          # parallel

OUTPUTS (per subject):
  fc_nilearn/<SID>_ts.npy   (307, T)
  fc_nilearn/<SID>_fcz.npy  (307, 307)
"""

from __future__ import annotations
import argparse, os, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance


# =============================================================================
# Paths and constants
# =============================================================================
ROOT          = Path("/home/violetz/CONN_revision")
ATLAS         = ROOT / "atlas" / "Schaefer300_Buckner7_2mm.nii.gz"
MANIFEST      = ROOT / "motion_audit" / "included_manifest.csv"
SUBJECTS_ROOT = Path("/nfs/turbo/lsa-tpolk/tpolk-mistorage/mind/subjects")
OUT_DIR       = ROOT / "fc_nilearn"                # <-- different folder from custom
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUNC_FILE = "dswaurun_01.nii"     # denoised, smoothed, warped, ART-regressed
N_ROIS    = 307


def find_func(sid: str) -> Path | None:
    """Find subject's denoised volume in placebo/ or placebodti/."""
    for sub in ("placebo", "placebodti"):
        p = SUBJECTS_ROOT / sid / sub / "func" / "connectivity" / "conn_proj" / "Func" / FUNC_FILE
        if p.exists():
            return p
    return None


def build_masker_and_conn():
    """
    Create the nilearn masker and connectivity estimator.
    Reused across all subjects in this worker (the atlas only needs to be loaded once).
    """
    masker = NiftiLabelsMasker(
        labels_img        = str(ATLAS),
        standardize       = False,       # CONN denoising already produced ready-to-use signal
        detrend           = False,       # already detrended in CONN
        low_pass          = None,        # already band-passed (0.008-0.09 Hz)
        high_pass         = None,
        background_label  = 0,
        strategy          = "mean",      # parcel value = mean of its voxels (textbook)
        resampling_target = None,        # atlas and functional already on same 2mm MNI grid
    ).fit()                              # caches the atlas once for re-use

    conn = ConnectivityMeasure(
        kind          = "correlation",
        cov_estimator = EmpiricalCovariance(),   # raw Pearson, no shrinkage (vs default LedoitWolf)
    )
    return masker, conn


def extract_one(sid: str, masker, conn) -> tuple[str, str]:
    """Run the full first-level extraction for one subject; skip if outputs exist."""
    ts_path  = OUT_DIR / f"{sid}_ts.npy"
    fcz_path = OUT_DIR / f"{sid}_fcz.npy"
    if ts_path.exists() and fcz_path.exists():
        return sid, "skip (already done)"

    func = find_func(sid)
    if func is None:
        return sid, f"FAIL: {FUNC_FILE} not found"

    # --- ROI timeseries via nilearn ---
    ts_TxN = masker.transform(str(func))     # nilearn convention: (T, N_ROIs)

    # --- Pearson correlation via nilearn (raw sample, no shrinkage) ---
    r = conn.fit_transform([ts_TxN])[0]      # (307, 307) Pearson r
    r = np.clip(r, -0.999999, 0.999999)
    fcz = np.arctanh(r).astype(np.float32)
    np.fill_diagonal(fcz, np.nan)            # self-correlation is meaningless

    # save in (N_ROIs, T) orientation to match the rest of our pipeline / downstream code
    ts = ts_TxN.T.astype(np.float32)
    np.save(ts_path, ts)
    np.save(fcz_path, fcz)
    T = ts.shape[1]
    return sid, f"OK  T={T}  off-diag |z| median={np.nanmedian(np.abs(fcz)):.3f}"


# ----------------------------------------------------------------------------
# Parallel runner — one shared masker per worker process
# ----------------------------------------------------------------------------
_MASKER_GLOBAL = None
_CONN_GLOBAL   = None


def _init_worker():
    global _MASKER_GLOBAL, _CONN_GLOBAL
    _MASKER_GLOBAL, _CONN_GLOBAL = build_masker_and_conn()


def _worker(sid):
    t1 = time.time()
    sid, msg = extract_one(sid, _MASKER_GLOBAL, _CONN_GLOBAL)
    return sid, msg, time.time() - t1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", help="subject IDs (default: read manifest)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="parallel processes (default: SLURM_CPUS_PER_TASK or 1)")
    args = ap.parse_args()

    print(f"Atlas       : {ATLAS}")
    print(f"Manifest    : {MANIFEST}")
    print(f"Output dir  : {OUT_DIR}   <-- separate from fc/ to keep both versions")
    print(f"Workers     : {args.workers}")
    print(f"Pipeline    : nilearn.NiftiLabelsMasker -> ConnectivityMeasure(EmpiricalCovariance) -> Fisher z\n")

    sids = args.subjects if args.subjects else pd.read_csv(MANIFEST)["subject_id"].tolist()
    t0 = time.time()

    if args.workers == 1:
        masker, conn = build_masker_and_conn()
        for i, sid in enumerate(sids, 1):
            t1 = time.time()
            sid, msg = extract_one(sid, masker, conn)
            print(f"  [{i:3d}/{len(sids)}] {sid:<10}  {msg}   ({time.time() - t1:.1f}s)", flush=True)
    else:
        with Pool(processes=args.workers, initializer=_init_worker) as pool:
            for i, (sid, msg, dt) in enumerate(pool.imap_unordered(_worker, sids), 1):
                print(f"  [{i:3d}/{len(sids)}] {sid:<10}  {msg}   ({dt:.1f}s)", flush=True)

    print(f"\nTotal elapsed: {(time.time() - t0)/60:.1f} min  "
          f"({len(sids)} subjects, {args.workers} workers)")


if __name__ == "__main__":
    main()
