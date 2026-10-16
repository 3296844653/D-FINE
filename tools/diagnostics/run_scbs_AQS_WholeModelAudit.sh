#!/usr/bin/env bash
# Existing checkpoint audit; detector extraction + independent linear probes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
audit_dry=0
audit_wait=0
audit_id='audit1'
audit_have_id=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) audit_dry=1 ;;
    --wait-gpu) audit_wait=1 ;;
    --help|-h) printf '用法：[--dry-run] [--wait-gpu] [audit1]\n'; exit 0 ;;
    --*) printf '未知选项：%s\n' "$1" >&2; exit 1 ;;
    *)
      [[ "$audit_have_id" == 0 ]] || { printf '只接受一个诊断编号。\n' >&2; exit 1; }
      audit_id="$1"; audit_have_id=1 ;;
  esac
  shift
done
[[ "$audit_id" =~ ^audit[1-9][0-9]*$ ]] || { printf '编号必须为audit1、audit2等。\n' >&2; exit 1; }
audit_run='/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1'
audit_checkpoint="${AQS_AUDIT_CHECKPOINT:-$audit_run/best_stg2.pth}"
audit_out="${AQS_AUDIT_OUTPUT_ROOT:-$(dirname "$audit_checkpoint")}/AQS_WholeModelAudit_${audit_id}"
audit_config='configs/diagnostics/scbs_AQS_WholeModelAudit.yml'
audit_conda="${AQS_AUDIT_CONDA:-${CONDA_EXE:-conda}}"
audit_env="${AQS_AUDIT_ENV:-wq}"
audit_gpu="${AQS_AUDIT_GPU:-0}"
[[ "$audit_gpu" =~ ^[0-9]+$ && "$audit_checkpoint" == /* && "$audit_out" == /* ]] || {
  printf 'GPU必须是单个编号；权重/输出必须是绝对路径。\n' >&2; exit 1;
}
[[ ! -e "$audit_out" && ! -L "$audit_out" && ! -e "$audit_out.zip" ]] || {
  printf '诊断输出或压缩包已存在，保留：%s；请改用audit2。\n' "$audit_out" >&2; exit 1;
}
for audit_dep in "$audit_config" tools/diagnostics/scbs_whole_model_audit.py \
  tools/diagnostics/scbs_feature_probe.py tools/diagnostics/scbs_encoder_query_coverage.py \
  tools/diagnostics/wait_for_gpu.py; do
  test -f "$audit_dep" || { printf '缺少文件：%s\n' "$audit_dep" >&2; exit 1; }
done
printf 'AQS WholeModelAudit：冻结原检测器，提取整个模型；训练3种子轻量分类探针。\n'
printf '权重：%s\n输出：%s\n' "$audit_checkpoint" "$audit_out"
[[ "$audit_wait" == 0 ]] || printf '自动等待模式：每30秒检查GPU，连续两次空闲后启动。\n'
if [[ "$audit_dry" == 1 ]]; then
  printf '演练：不连接GPU、不提取数据、不创建目录；正式运行去掉--dry-run。\n'; exit 0
fi
command -v "$audit_conda" >/dev/null || { printf '找不到Conda。\n' >&2; exit 1; }
audit_log="${audit_out}_console.log"
[[ ! -e "$audit_log" ]] || { printf '控制台日志已存在，请使用新编号：%s\n' "$audit_log" >&2; exit 1; }
test -d "$(dirname "$audit_out")" || { printf '输出父目录不存在。\n' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$audit_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
audit_gpu_options=(--gpu "$audit_gpu")
[[ "$audit_wait" == 0 ]] || audit_gpu_options+=(--wait)
"$audit_conda" run --no-capture-output -n "$audit_env" python -u \
  tools/diagnostics/wait_for_gpu.py "${audit_gpu_options[@]}"
# Recheck result paths after a potentially long wait before opening any logs.
[[ ! -e "$audit_out" && ! -L "$audit_out" && ! -e "$audit_out.zip" && ! -e "$audit_log" ]] || {
  printf '等待期间已有相同编号输出，保留结果并停止：%s\n' "$audit_out" >&2; exit 1;
}
if ! (set -o noclobber; : > "$audit_log") 2>/dev/null; then
  printf '无法新建控制台日志，保留已有文件：%s\n' "$audit_log" >&2; exit 1
fi
"$audit_conda" run --no-capture-output -n "$audit_env" python -u \
  tools/diagnostics/scbs_whole_model_audit.py --config "$audit_config" \
  --checkpoint "$audit_checkpoint" --output-dir "$audit_out" --device cuda --mode all \
  2>&1 | tee -a "$audit_log"
printf '\n诊断完成：%s\n压缩包：%s.zip\n' "$audit_out" "$audit_out"
