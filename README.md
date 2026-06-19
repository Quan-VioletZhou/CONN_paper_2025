# CONN_paper_2025

Code for the manuscript **"Age-related Changes in Brain Connectivity: Cross-sectional and Longitudinal Effects"**
(*Imaging Neuroscience*, IMAG-26-0103).

The pipeline takes resting-state fMRI data through preprocessing in MATLAB
(CONN toolbox), builds a functionally defined parcellation, extracts first-level
ROI-to-ROI connectivity in Python (nilearn), performs motion / quality control,
and fits linear mixed-effects models in R.

---

## Repository layout

### Preprocessing — MATLAB + CONN toolbox

| File | Purpose |
|---|---|
| `run_conn.m` | Interactive wrapper to preprocess one subject through CONN. |
| `run_conn_pipeline.m` | CONN batch pipeline: SPM-default MNI preprocessing → ART scrubbing (FD > 0.45 mm, BOLD > 5 SD, previous-frame flag) → denoising (5 WM + 5 CSF aCompCor + Friston-24 + spike regression + band-pass 0.008–0.09 Hz). |
| `run_all_subjects.sh` | Bash wrapper looping over all `mind*` subjects non-interactively. |
| `QualityControl_2025.mlx` | MATLAB live notebook for distribution-based QC. |
| `conn_mindo_results.mlx`, `conn_result_single_sub.mlx` | Live notebooks for inspecting CONN output. |

### Brain parcellation — Python

| File | Purpose |
|---|---|
| `build_merged_atlas.py` | Builds the **Schaefer-300 (Yeo-7 cortex) + Buckner-7 (cerebellum) → 307-ROI, 8-network** atlas used for all connectivity estimates. |
| `validate_atlas.py` | Validation / QC of the merged atlas (ROI counts, coverage, network assignments). |

### First-level FC extraction — Python (nilearn)

| File | Purpose |
|---|---|
| `extract_roi_timeseries_nilearn.py` | Applies the merged atlas to each subject's CONN-denoised volume (`dswaurun_01.nii`). Uses `nilearn.maskers.NiftiLabelsMasker` + `nilearn.connectome.ConnectivityMeasure` (empirical covariance, no shrinkage) → Fisher-z. Saves per subject: `<sid>_ts.npy` (307 × T) and `<sid>_fcz.npy` (307 × 307). Parallelized via `multiprocessing` / `$SLURM_CPUS_PER_TASK`. |
| `compute_network_fc.py` | Collapses each 307 × 307 matrix into network-level measures: within-network FC, between-network FC (two-step, equal weight per network pair), and segregation `(within − between)/within`. |
| `build_neuro_mastersheet.py` | Assembles the analysis master sheet (network measures + cohort + motion + clinical status). |

### Motion / quality control — Python

| File | Purpose |
|---|---|
| `cohort_motion_audit.py` | Per-subject FD / BOLD signal-quality audit from ART outputs. |
| `distribution_qc.py` | Distribution-based QC: flags participants > 3 SD from cohort mean on any motion/signal metric. |
| `final_manifest.py` | Locks participant inclusion using the ≥ 50% high-quality-volume threshold and distribution QC. |
| `qcfc_analysis.py` | Quality-control–functional-connectivity (QC-FC) analyses (Power et al., 2012; Ciric et al., 2018): network-level QC-FC (FD vs within/between/segregation) and edge-level QC-FC with distance dependence. |

### Statistical modelling — R and Python

| File | Purpose |
|---|---|
| `run_models_connbx_6p3.py` | Cross-sectional (Model 1, YA + W1) and longitudinal (Model 2, YA + W1 + W2) mixed-effects models for within-network connectivity, with sum-to-zero network effect coding and mean FD covariate. |
| `followup_per_network_cross_Tier2.R` | Cross-sectional per-network follow-up regressions (`nlme`). |
| `followup_per_network_Tier2.R` | Longitudinal per-network follow-up mixed-effects models (`nlme`), with model-fit comparison. |
| `plot_measures_YA_OA.R` | Distribution / outlier plots of connectivity measures by age group. |
| `CONN_LMM.Rmd` | Primary linear mixed-effects models (cross-sectional + longitudinal; within / between / segregation). |
| `Paper_Plot_for_submission.Rmd` | Figure generation for the manuscript (ggplot2). |

---

## Pipeline order

```
                     Raw fMRI per subject
                              │
        MATLAB + CONN  (run_conn.m → run_conn_pipeline.m)
                              │
   Preprocessing (default_mni) + ART (FD>0.45, BOLD>5SD, prev-frame, first-4 dummy)
                              │
   Denoising (5WM/5CSF aCompCor, Friston-24, spike regression, band-pass 0.008–0.09 Hz)
                              │
                  dswaurun_01.nii  (denoised, MNI 2 mm)
                              │
        Atlas (build_merged_atlas.py): Schaefer-300 + Buckner-7 → 307 ROIs
                              │
        Python + nilearn (extract_roi_timeseries_nilearn.py) → fcz.npy
                              │
        Network measures (compute_network_fc.py → build_neuro_mastersheet.py)
                              │
        Motion / QC (cohort_motion_audit.py, distribution_qc.py,
                     final_manifest.py, qcfc_analysis.py)
                              │
        R / Python mixed-effects models (CONN_LMM.Rmd, run_models_connbx_6p3.py,
                     followup_per_network_*.R)
```

---

## Dependencies

- **MATLAB R2022b+** with **CONN toolbox 2022** and **SPM12**
- **Python 3.11+**: `nilearn ≥ 0.10`, `scikit-learn ≥ 1.1`, `nibabel ≥ 5`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`
- **R 4.4+**: `nlme`, `lme4`, `lmerTest`, `dplyr`, `tidyr`, `ggplot2`

---

## Quick start (Python extraction)

```bash
# extract one subject (testing)
python3 extract_roi_timeseries_nilearn.py mindo100

# extract all subjects in the manifest, 16 workers in parallel
python3 extract_roi_timeseries_nilearn.py --workers 16
```

The script auto-detects `$SLURM_CPUS_PER_TASK` and skips subjects whose outputs
already exist (runs are resumable).

---

## Citation

If you use this code, please cite the manuscript and the underlying tools:

- **CONN toolbox** — Whitfield-Gabrieli S, Nieto-Castanon A. *Brain Connect.* 2012;2:125–141.
- **Schaefer parcellation** — Schaefer A, Kong R, Gordon EM, et al. *Cereb Cortex.* 2018;28:3095–3114.
- **Yeo 7-network parcellation** — Yeo BTT, Krienen FM, Sepulcre J, et al. *J Neurophysiol.* 2011;106:1125–1165.
- **Buckner cerebellar atlas** — Buckner RL, Krienen FM, Castellanos A, Diaz JC, Yeo BTT. *J Neurophysiol.* 2011;106:2322–2345.
- **nilearn** — Abraham A, Pedregosa F, Eickenberg M, et al. *Front Neuroinform.* 2014;8:14.
- **QC-FC / motion benchmarks** — Power JD, et al. *NeuroImage.* 2012;59:2142–2154; Ciric R, et al. *Nat Protoc.* 2018;13:2801–2826.

---

## Contact

Quan ("Violet") Zhou — Polk Lab, University of Michigan
