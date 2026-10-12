#!/usr/bin/env bash
# Six frozen validation passes: Run2 + five SAME-checkpoint SADF interventions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
audit_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
[[ "$#" -le 1 ]] || audit_fail '用法：bash tools/diagnostics/run_scbs_SADF_runtime_audit.sh [新的绝对输出目录]'
audit_root='/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw'
audit_output="${1:-$audit_root/dfine_s_scbs_hrw_epochs110_SADF_seed0_run1/SADF_runtime_audit_run1}"
audit_checkpoint="${SCBS_SADF_AUDIT_CHECKPOINT:-$audit_root/dfine_s_scbs_hrw_epochs110_SADF_seed0_run1/best_stg2.pth}"
audit_baseline="${SCBS_SADF_AUDIT_RUN2_CHECKPOINT:-$audit_root/dfine_s_scbs_hrw_epochs110_seed0_run2/best_stg2.pth}"
audit_conda="${SCBS_SADF_AUDIT_CONDA:-${CONDA_EXE:-conda}}"
audit_env="${SCBS_SADF_AUDIT_ENV:-wq}"
audit_gpu="${SCBS_SADF_AUDIT_GPU:-0}"
[[ "$audit_gpu" =~ ^[0-9]+$ && "$audit_output" == /* ]] || audit_fail 'GPU须为单个非负整数，输出须为绝对路径'
[[ "${audit_output##*/}" == *SADF* ]] || audit_fail '诊断输出名称须包含 SADF'
[[ ! -e "$audit_output" && ! -L "$audit_output" ]] || audit_fail "保留旧诊断结果，请指定新目录：$audit_output"
audit_console="${audit_output}.console.log"
[[ ! -e "$audit_console" && ! -L "$audit_console" ]] || audit_fail "保留旧日志：$audit_console"
test -f "$audit_checkpoint" || audit_fail "缺少SADF最佳权重：$audit_checkpoint"
test -f "$audit_baseline" || audit_fail "缺少Run2最佳权重：$audit_baseline"
command -v "$audit_conda" >/dev/null 2>&1 || audit_fail '找不到Conda'
export CUDA_VISIBLE_DEVICES="$audit_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
printf '只进行验证，不训练、不改权重、不占用torchrun端口。请确认GPU空闲。\n'
mkdir -p -- "$(dirname "$audit_output")"
"$audit_conda" run --no-capture-output -n "$audit_env" \
python -u tools/diagnostics/audit_sadf_runtime.py \
  --checkpoint "$audit_checkpoint" --baseline-checkpoint "$audit_baseline" \
  --output-dir "$audit_output" --device cuda --source ema --batch-size 8 --workers 4 \
  2>&1 | tee "$audit_console"
test -s "$audit_output/complete.json"
mv -- "$audit_console" "$audit_output/console.log"
printf '\nSADF内部诊断完成：%s\n' "$audit_output"
printf '先看 SADF_runtime_report.md、SADF_runtime_summary.json 和 SADF_vs_Run2_paired_groups.json。\n'
