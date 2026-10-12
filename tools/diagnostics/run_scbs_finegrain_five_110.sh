#!/usr/bin/env bash
# Five independent full runs on ONE GPU; no loss changes, resume or deletion.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
fg_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
fg_dry=0; fg_modes=(); fg_seen=' '
for fg_arg in "$@"; do
  case "$fg_arg" in
    --dry-run) fg_dry=1 ;;
    -h|--help)
      printf '%s\n' '用法：bash tools/diagnostics/run_scbs_finegrain_five_110.sh [--dry-run] [模块名 ...]'
      printf '%s\n' '默认独立运行5组，110轮/停止增强100/batch32/seed0；默认导出诊断并和Run2对照。'
      printf '%s\n' 'FINEGRAIN_RUN_ID=2改运行编号；FINEGRAIN_EXPORT=0只训练；FINEGRAIN_GPU=0指定GPU。'
      exit 0 ;;
    DetailOnlyCrossGate|WindowInteractionP3|BoxScaleContrastL2|PartInteractionL2|PairEvidenceHead)
      case "$fg_seen" in *" $fg_arg "*) fg_fail "重复模块：$fg_arg" ;; esac
      fg_seen+="$fg_arg "; fg_modes+=("$fg_arg") ;;
    *) fg_fail "未知参数/模块：$fg_arg" ;;
  esac
done
[[ ${#fg_modes[@]} -gt 0 ]] || fg_modes=(BoxScaleContrastL2 PartInteractionL2 DetailOnlyCrossGate WindowInteractionP3 PairEvidenceHead)
fg_root="${FINEGRAIN_OUTPUT_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
fg_run="${FINEGRAIN_RUN_ID:-1}"; fg_gpu="${FINEGRAIN_GPU:-0}"; fg_export="${FINEGRAIN_EXPORT:-1}"
fg_conda="${FINEGRAIN_CONDA:-${CONDA_EXE:-conda}}"; fg_env="${FINEGRAIN_CONDA_ENV:-wq}"
fg_run2="${FINEGRAIN_RUN2_CHECKPOINT:-$fg_root/dfine_s_scbs_hrw_epochs110_seed0_run2/best_stg2.pth}"
fg_queue="$fg_root/finegrain_five_epochs110_seed0_run${fg_run}"
[[ "$fg_root" == /* && "$fg_run" =~ ^[1-9][0-9]*$ && "$fg_gpu" =~ ^[0-9]+$ && "$fg_export" =~ ^[01]$ ]] || fg_fail '检查输出绝对路径、RUN_ID、GPU和EXPORT'
[[ ! -e "$fg_queue" && ! -L "$fg_queue" ]] || fg_fail "队列目录已存在：$fg_queue；请更改FINEGRAIN_RUN_ID"
for fg_mode in "${fg_modes[@]}"; do
  fg_config="configs/dfine/dfine_s_scbs_hrw_110_${fg_mode}.yml"
  fg_out="$fg_root/dfine_s_scbs_hrw_epochs110_${fg_mode}_seed0_run${fg_run}"
  [[ -f "$fg_config" && -f train.py ]] || fg_fail "缺少文件：$fg_config"
  [[ ! -e "$fg_out" && ! -L "$fg_out" ]] || fg_fail "结果目录已存在，未启动任何训练：$fg_out"
  printf '%s\n  配置：%s\n  输出：%s\n' "$fg_mode" "$fg_config" "$fg_out"
done
fg_python=("$fg_conda" run --no-capture-output -n "$fg_env" python)
if [[ "$fg_dry" == 1 && -n "${FINEGRAIN_PYTHON:-}" ]]; then
  fg_python=("$FINEGRAIN_PYTHON")
else
  command -v "$fg_conda" >/dev/null 2>&1 || fg_fail '找不到Conda'
fi
export CUDA_VISIBLE_DEVICES="$fg_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
if [[ "$fg_dry" == 1 ]]; then
  "${fg_python[@]}" tools/diagnostics/scbs_finegrain_queue.py preflight --dry-run "${fg_modes[@]}"
  printf '演练通过；未训练、未创建目录。\n'; exit 0
fi
fg_check=(tools/diagnostics/scbs_finegrain_queue.py preflight "${fg_modes[@]}")
[[ "$fg_export" == 0 ]] || fg_check+=(--run2-checkpoint "$fg_run2")
"${fg_python[@]}" "${fg_check[@]}"
if command -v flock >/dev/null 2>&1; then
  exec 9>"/tmp/dfine-finegrain-gpu${fg_gpu}.lock"
  flock -n 9 || fg_fail '同GPU的另一份finegrain队列仍在运行'
fi
fg_check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || fg_fail '无法确认GPU状态'
  local fg_pids fg_pid fg_process
  fg_pids="$(nvidia-smi -i "$fg_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" || fg_fail 'GPU查询失败'
  while IFS= read -r fg_pid; do
    fg_pid="${fg_pid//[[:space:]]/}"
    [[ "$fg_pid" =~ ^[0-9]+$ ]] || continue
    fg_process="$(ps -ww -p "$fg_pid" -o args= 2>/dev/null || true)"
    [[ -n "$fg_process" ]] || continue
    case "$fg_process" in *Xorg*|*gnome-shell*|*gnome-remote-desktop*) continue ;; esac
    fg_fail "GPU$fg_gpu仍有进程$fg_pid：$fg_process；没有终止它"
  done <<< "$fg_pids"
}
fg_check_gpu
"${fg_python[@]}" tools/diagnostics/test_finegrain_trials.py FinegrainTests.test_cuda_fp16_all_five
mkdir -p -- "$fg_root"
mkdir -- "$fg_queue"
trap 'printf "队列中断，所有已有结果保留；没有自动恢复训练。\n" >&2; exit 130' INT TERM

fg_evaluate() {
  local fg_cfg="$1" fg_checkpoint="$2" fg_directory="$3"
  "$fg_conda" run --no-capture-output -n "$fg_env" \
    torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
    train.py -c "$fg_cfg" -r "$fg_checkpoint" --test-only \
    --output-dir "$fg_directory/evaluation" --use-amp --seed=0 \
    -u export_confusion_matrix=true export_query_diagnostics=true \
    query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
    2>&1 | tee "$fg_directory/evaluation_console.log"
  "${fg_python[@]}" tools/diagnostics/analyze_detection_diagnostics.py \
    --input-dir "$fg_directory/evaluation" --output-dir "$fg_directory/analysis" \
    2>&1 | tee "$fg_directory/analysis_console.log"
}
# Re-evaluate Run2 ONCE in a new directory; never retrain or modify it.
fg_reference="$fg_queue/BaselineRun2_diagnostics_eval"
if [[ "$fg_export" == 1 ]]; then
  mkdir -- "$fg_reference"
  fg_evaluate configs/dfine/dfine_s_scbs_hrw_110.yml "$fg_run2" "$fg_reference"
fi
for fg_mode in "${fg_modes[@]}"; do
  fg_check_gpu
  fg_config="configs/dfine/dfine_s_scbs_hrw_110_${fg_mode}.yml"
  fg_out="$fg_root/dfine_s_scbs_hrw_epochs110_${fg_mode}_seed0_run${fg_run}"
  mkdir -- "$fg_out"
  "${fg_python[@]}" tools/diagnostics/scbs_finegrain_queue.py snapshot "$fg_mode" "$fg_out"
  printf '\n开始完整训练：%s（110轮，独立运行，不叠加AQS/损失）\n' "$fg_mode"
  "$fg_conda" run --no-capture-output -n "$fg_env" \
    torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1 \
    train.py -c "$fg_config" --output-dir "$fg_out" --use-amp --seed=0 \
    2>&1 | tee "$fg_out/console.log"
  [[ -f "$fg_out/best_stg2.pth" ]] || fg_fail "$fg_mode 未产生best_stg2，保留输出并停止队列"
  if [[ "$fg_export" == 1 ]]; then
    mkdir -- "$fg_out/diagnostics_eval"
    fg_evaluate "$fg_config" "$fg_out/best_stg2.pth" "$fg_out/diagnostics_eval"
    "${fg_python[@]}" tools/diagnostics/scbs_finegrain_queue.py compare "$fg_out" "$fg_reference"
  fi
  "${fg_python[@]}" tools/diagnostics/scbs_finegrain_queue.py record "$fg_mode" "$fg_out" "$fg_queue"
done
printf '\n全部完成。汇总：%s/results.jsonl\n' "$fg_queue"
