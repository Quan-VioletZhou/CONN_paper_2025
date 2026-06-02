# CONN_paper_2025

Code for the manuscript **"Age-related Changes in Brain Connectivity: Cross-sectional and Longitudinal Effects"**
(*Imaging Neuroscience*, IMAG-26-0103).

The pipeline takes resting-state fMRI data through preprocessing in MATLAB
(CONN toolbox), first-level ROI extraction in Python (nilearn + Schaefer-300
+ Buckner-7 atlas), and statistical modelling in R (linear mixed-effects).

---

## Repository layout

### Preprocessing — MATLAB + CONN toolbox

| File | Purpose |
|---|---|
| `run_conn.m` | Interactive wrapper to preprocess one subject through CONN (asks for subject ID, builds paths, hands off to `run_conn_pipeline`). |
| `run_conn_pipeline.m` | The actual CONN batch pipeline: SPM-default MNI preprocessing → ART scrubbing (FD > 0.45 mm, BOLD > 5 SD, previous-frame flag) → denoising (5 WM + 5 CSF aCompCor + Friston-24 + spike regression + band-pass 0.008–0.09 Hz) → ROI-to-ROI first-level for the original CONN HCP-ICA atlas. |
| `run_all_subjects.sh` | Bash wrapper that loops over all `mind*` subject folders and calls `run_conn.m` non-interactively; logs per-subject output. |
| `QualityControl_2025.mlx` | MATLAB live notebook for distribution-based QC (mean/SD of BOLD and FD; >3 SD from group flagged for review). |
| `conn_mindo_results.mlx`, `conn_result_single_sub.mlx` | Live notebooks for inspecting CONN output for one subject / one cohort. |

### First-level FC extraction — Python (new atlas)

| File | Purpose |
|---|---|
| `extract_roi_timeseries_nilearn.py` | Applies the **Schaefer-300 (Yeo-7 cortex) + Buckner-7 (cerebellum) → 8 networks** atlas to each subject's CONN-denoised volume (`dswaurun_01.nii`). Saves per subject: `<sid>_ts.npy` (307 × T) and `<sid>_fcz.npy` (307 × 307 Fisher z). Uses `nilearn.maskers.NiftiLabelsMasker` + `nilearn.connectome.ConnectivityMeasure(cov_estimator=EmpiricalCovariance)` → raw sample Pearson r, Fisher z-transformed. Supports parallel execution via `multiprocessing.Pool` and `--workers` (or `$SLURM_CPUS_PER_TASK`). |

### Statistical modelling — R

| File | Purpose |
|---|---|
| `CONN_LMM.Rmd` | Linear mixed-effects models (nlme): cross-sectional (Model 1, YA + W1) and longitudinal (Model 2, W1 + W2) within-network / between-network / segregation connectivity, with random intercepts per subject and sum-to-zero effect coding for networks. Includes mean FD as a per-wave covariate. |
| `Paper_Plot_for_submission.Rmd` | Figure generation for the manuscript and revision (ggplot2). |

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
                  dswaurun_01.nii  (denoised, MNI 2 mm, 240 vols)
                              │
        Python + nilearn  (extract_roi_timeseries_nilearn.py)
                              │
        Schaefer-300 + Buckner-7 atlas → 307 ROIs
                              │
              ts.npy (307 × T)   fcz.npy (307 × 307 Fisher z)
                              │
        R + nlme  (CONN_LMM.Rmd, Paper_Plot_for_submission.Rmd)
                              │
        Linear mixed-effects models + figures
```

---

## Dependencies

- **MATLAB R2022b+** with the **CONN toolbox 2022** and **SPM12**
- **Python 3.11+** with:
  - `nilearn ≥ 0.10`
  - `scikit-learn ≥ 1.1`
  - `nibabel ≥ 5`
  - `numpy`, `pandas`
- **R 4.4+** with:
  - `nlme`, `lme4`, `lmerTest`
  - `dplyr`, `tidyr`, `ggplot2`, `forcats`, `viridis`, `RColorBrewer`

---

## Quick start (Python extraction)

```bash
# extract one subject (testing)
python3 extract_roi_timeseries_nilearn.py mindo100

# extract all subjects listed in a manifest (default), 16 workers in parallel
python3 extract_roi_timeseries_nilearn.py --workers 16
```

The script auto-detects `$SLURM_CPUS_PER_TASK` if running under SLURM, and
skips subjects whose outputs already exist (so runs are resumable).

---

## Citation

If you use this code, please cite the manuscript and the underlying tools:

- **CONN toolbox** — Whitfield-Gabrieli S, Nieto-Castanon A. *Brain Connect.* 2012;2:125–141.
- **Schaefer parcellation** — Schaefer A, Kong R, Gordon EM, et al. *Cereb Cortex.* 2018;28:3095–3114.
- **Buckner cerebellar atlas** — Buckner RL, Krienen FM, Castellanos A, Diaz JC, Yeo BTT. *J Neurophysiol.* 2011;106:2322–2345.
- **nilearn** — Abraham A, Pedregosa F, Eickenberg M, et al. *Front Neuroinform.* 2014;8:14.

---

## Contact

Quan ("Violet") Zhou — Polk Lab, University of Michigan
