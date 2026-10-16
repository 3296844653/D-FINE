#!/usr/bin/env bash
# Existing best EMA -> ON/OFF validation + diagnostics. NEVER train/resume-fit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
der_export_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
der_export_dry=0
if [[ "${1:-}" == --dry-run ]]; then der_export_dry=1; shift; fi
[[ "$#" -le 2 ]] || der_export_fail '用法：[--dry-run] [run1|run2...] [export1|export2...]'
der_export_run="${1:-run1}"; der_export_id="${2:-export1}"
[[ "$der_export_run" =~ ^run[1-9][0-9]*$ && "$der_export_id" =~ ^export[1-9][0-9]*$ ]] || der_export_fail '运行/导出编号无效'
der_export_root="${SCBS_DER_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
der_export_training="$der_export_root/dfine_s_scbs_hrw_epochs110_AQS_DiscriminativeEvidenceL2_seed0_${der_export_run}"
der_export_checkpoint="${SCBS_DER_CHECKPOINT:-$der_export_training/best_stg2.pth}"
der_export_bundle="$der_export_training/DiscriminativeEvidenceL2_diagnostics_${der_export_id}"
der_export_config='configs/dfine/dfine_s_scbs_hrw_110_AQS_DiscriminativeEvidenceL2.yml'
der_export_helper='tools/diagnostics/scbs_discriminative_evidence_export.py'
der_export_experiment='tools/diagnostics/scbs_discriminative_evidence_experiment.py'
der_export_wrapper='tools/diagnostics/eval_discriminative_evidence.py'
der_export_conda="${SCBS_DER_CONDA:-${CONDA_EXE:-conda}}"; der_export_env="${SCBS_DER_ENV:-wq}"
der_export_gpu="${SCBS_DER_GPU:-0}"; der_export_port="${SCBS_DER_PORT:-7781}"
[[ "$der_export_root" == /* && "$der_export_checkpoint" == /* ]] || der_export_fail '根目录和权重须为绝对路径'
[[ "$der_export_gpu" =~ ^[0-9]+$ && "$der_export_port" =~ ^[0-9]+$ ]] || der_export_fail 'GPU/端口须为整数'
[[ "$der_export_port" -ge 1024 && "$der_export_port" -le 65535 ]] || der_export_fail '端口须为1024..65535'
test -s "$der_export_checkpoint" || der_export_fail "缺少非空最佳权重：$der_export_checkpoint"
test -d "$der_export_training" || der_export_fail '缺少原训练目录'
[[ ! -e "$der_export_bundle" && ! -L "$der_export_bundle" ]] || der_export_fail "保留旧导出，请改用export2等新编号：$der_export_bundle"
for der_export_file in train.py "$der_export_config" "$der_export_helper" "$der_export_experiment" "$der_export_wrapper" \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  test -f "$der_export_file" || der_export_fail "缺少源码/工具：$der_export_file"
done
printf '只导出已有最佳EMA，不训练。\n权重：%s\n新目录：%s\nON/OFF两次均保留AQS-L2 R1。\n' "$der_export_checkpoint" "$der_export_bundle"
if [[ "$der_export_dry" == 1 ]]; then
  printf '仅演练：不创建目录、不加载GPU、不连接服务器。\n'; exit 0
fi
command -v "$der_export_conda" >/dev/null 2>&1 || der_export_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$der_export_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
"$der_export_conda" run --no-capture-output -n "$der_export_env" python - "$der_export_gpu" "$der_export_port" <<'PY'
import socket, subprocess, sys
from pathlib import Path
rows = subprocess.run(['nvidia-smi','-i',sys.argv[1],'--query-compute-apps=pid,process_name',
    '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
desktop = {'Xorg','gnome-shell','gnome-remote-desktop-daemon'}
busy = [r for r in rows.splitlines() if Path(r.split(',',1)[-1].strip()).name not in desktop]
assert not busy, f'GPU被占用，未导出；不结束任何项目：{busy}'
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
    s.bind(('0.0.0.0',int(sys.argv[2])))
print('GPU/端口检查通过。',flush=True)
PY
"$der_export_conda" run --no-capture-output -n "$der_export_env" python -u "$der_export_helper" prepare "$der_export_checkpoint" "$der_export_bundle"
trap 'der_export_status=$?; printf "导出未完成，退出码=%s；保留部分结果：%s\n" "$der_export_status" "$der_export_bundle" >&2; exit "$der_export_status"' ERR
"$der_export_conda" run --no-capture-output -n "$der_export_env" python -u tools/diagnostics/export_aqs_parameters.py \
--checkpoint "$der_export_checkpoint" --temperature .10 --output "$der_export_bundle/aqs_parameters.json"
"$der_export_conda" run --no-capture-output -n "$der_export_env" python -u "$der_export_experiment" parameters \
"$der_export_checkpoint" "$der_export_bundle/DiscriminativeEvidenceL2_parameters.json"
for der_export_mode in on off; do
  der_export_diag="$der_export_bundle/diagnostics_eval"; der_export_flag=true
  if [[ "$der_export_mode" == off ]]; then
    der_export_diag="$der_export_bundle/DiscriminativeEvidenceL2_off_eval"; der_export_flag=false
  fi
  printf '\n验证模式=%s；只切换新模块，保留同一EMA和AQS。\n' "$der_export_mode"
  "$der_export_conda" run --no-capture-output -n "$der_export_env" \
  torchrun --master_port="$der_export_port" --nproc_per_node=1 --no-python python -u "$der_export_wrapper" \
  --mode "$der_export_mode" -c "$der_export_config" -r "$der_export_checkpoint" --test-only \
  --output-dir "$der_export_diag/evaluation" --use-amp --seed=0 \
  -u export_confusion_matrix=true export_query_diagnostics=true \
  query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
  diagnostic_DiscriminativeEvidenceL2_output_enabled="$der_export_flag" \
  2>&1 | tee "$der_export_bundle/logs/${der_export_mode}_evaluation.log"
  "$der_export_conda" run --no-capture-output -n "$der_export_env" python -u tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$der_export_diag/evaluation" --output-dir "$der_export_diag/analysis" \
  2>&1 | tee "$der_export_bundle/logs/${der_export_mode}_analysis.log"
  test -s "$der_export_diag/analysis/summary.json"
  test -s "$der_export_diag/evaluation/confusion_matrix/confusion_matrix.json"
done
"$der_export_conda" run --no-capture-output -n "$der_export_env" python -u "$der_export_experiment" on-off "$der_export_bundle"
"$der_export_conda" run --no-capture-output -n "$der_export_env" python -u "$der_export_helper" audit "$der_export_bundle"

# Read only the two known reference runs. Do not choose unrelated recent runs.
# Prefer complete modern exports; older AQS missing size tables remains valid.
der_export_find_reference() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate/analysis/summary.json" && -f "$candidate/analysis/gt_records.jsonl" && -f "$candidate/evaluation/query_diagnostics/metadata.json" ]]; then
      printf '%s' "$candidate"; return 0
    fi
  done
}
der_export_aqs_run="$der_export_root/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1"
der_export_run2_run="$der_export_root/dfine_s_scbs_hrw_epochs110_seed0_run2"
der_export_aqs_reference="${SCBS_DER_AQS_REFERENCE:-$(der_export_find_reference "$der_export_aqs_run/diagnostics_eval_v2" "$der_export_aqs_run/diagnostics_eval")}"
der_export_run2_reference="${SCBS_DER_RUN2_REFERENCE:-$(der_export_find_reference "$der_export_run2_run/diagnostics_export1" "$der_export_run2_run/diagnostics_eval")}"
for der_export_label in AQS_L2_R1 BaselineRun2; do
  der_export_reference="$der_export_aqs_reference"
  [[ "$der_export_label" != BaselineRun2 ]] || der_export_reference="$der_export_run2_reference"
  if [[ -n "$der_export_reference" ]]; then
    "$der_export_conda" run --no-capture-output -n "$der_export_env" python -u "$der_export_experiment" compare \
    "$der_export_bundle" "$der_export_reference" "$der_export_label" \
    2>&1 | tee "$der_export_bundle/logs/compare_${der_export_label}.log"
  else
    printf '未找到%s完整诊断；新结果已保存，未伪造参考比较。\n' "$der_export_label"
  fi
done
"$der_export_conda" run --no-capture-output -n "$der_export_env" python -u "$der_export_helper" package "$der_export_bundle"
printf '\n导出完成：%s\n未修改训练权重、日志、模型、配置或数据。\n' "$der_export_bundle"
