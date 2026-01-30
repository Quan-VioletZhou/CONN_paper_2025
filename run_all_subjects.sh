#!/usr/bin/env bash
# run_all_subjects.sh
# Runs your existing prepare_conn_pipeline.m for all subjects mind[a-z][0-9]{3}
# - No edits to your MATLAB code required
# - Feeds subject ID to the `input()` prompt and answers "y" to overwrite
# - Logs per-subject output to ./logs/<subject>.log
# - Summarizes failures in ./conn_subject_errors.log

set -u
ROOT_DIR="/nfs/tpolk/mind/subjects"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PIPELINE_DIR/logs"
ERROR_LOG="$PIPELINE_DIR/conn_subject_errors.log"
MATLAB_BIN="${MATLAB_BIN:-matlab}"   # allow override: MATLAB_BIN=/path/to/matlab ./run_all_subjects.sh

mkdir -p "$LOG_DIR"
: > "$ERROR_LOG"  # truncate

echo "🔄 Starting CONN pipeline for all subjects in $ROOT_DIR"
echo "📜 MATLAB scripts path: $PIPELINE_DIR"
echo "🗂  Logs in: $LOG_DIR"
echo "🧾 Error summary: $ERROR_LOG"
echo "----------------------------------------------"

shopt -s nullglob
total=0
ok=0
fail=0

for subj_path in "$ROOT_DIR"/mind*; do
  [[ -d "$subj_path" ]] || continue
  subject_id="$(basename "$subj_path")"
  # bash regex: mind + one letter + 3 digits
  #  if [[ ! "$subject_id" =~ ^mind[a-z][0-9]{3}$ ]]; then
  if [[ ! "$subject_id" =~ ^mindo[0-9]{3}$ ]]; then
    continue
  fi

  (( total++ ))
  echo "🚀 Running subject: $subject_id"

  subj_log="$LOG_DIR/${subject_id}.log"

  # Feed two lines to MATLAB:
  #   1) subject_id  (for the first input prompt)
  #   2) y           (auto-confirm overwrite if meta row exists)
  printf "%s\ny\n" "$subject_id" | \
  "$MATLAB_BIN" -nodisplay -nosplash -nodesktop -r \
    "try, addpath('$PIPELINE_DIR'); run_conn; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);" \
    > "$subj_log" 2>&1

  rc=$?

  # Success check: also verify expected CONN result file exists (either placebo or placebodti)
  roi1="$ROOT_DIR/$subject_id/placebo/func/connectivity/conn_proj/conn_project01/results/firstlevel/SBC_01/resultsROI_Subject001_Condition001.mat"
  roi2="$ROOT_DIR/$subject_id/placebodti/func/connectivity/conn_proj/conn_project01/results/firstlevel/SBC_01/resultsROI_Subject001_Condition001.mat"
  if [[ $rc -eq 0 && ( -f "$roi1" || -f "$roi2" ) ]]; then
    (( ok++ ))
    echo "✅ Completed: $subject_id"
  else
    (( fail++ ))
    echo "❌ Failed: $subject_id"
    {
      echo "===== $subject_id ====="
      echo "Command exit code: $rc"
      echo "--- tail of log ---"
      tail -n 150 "$subj_log" || true
      echo
    } >> "$ERROR_LOG"
  fi

  echo "----------------------------------------------"
done

echo "🎉 Done. Total: $total | ✅ Success: $ok | ❌ Failed: $fail"
echo "🧾 See: $ERROR_LOG"
echo "📂 Logs: $LOG_DIR"

