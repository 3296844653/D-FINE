#!/usr/bin/env bash
# Three INDEPENDENT AQS + one-module experiments, sequential on one GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
evidence_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
evidence_dry=0
if [[ "${1:-}" == --dry-run ]]; then evidence_dry=1; shift; fi
[[ "$#" -le 2 ]] || evidence_fail '用法：bash tools/diagnostics/run_scbs_AQS_instance_evidence_110.sh [--dry-run] [all|模块名] [run1|run2...]'
evidence_selection="${1:-all}"; evidence_run="${2:-run1}"
[[ "$evidence_run" =~ ^run[1-9][0-9]*$ ]] || evidence_fail '运行编号须为run1/run2等'
case "$evidence_selection" in
  all) evidence_modes=(PartEvidenceSelection TrilinearDetailSampling LocalInteractionEvidence) ;;
  PartEvidenceSelection|TrilinearDetailSampling|LocalInteractionEvidence) evidence_modes=("$evidence_selection") ;;
  *) evidence_fail '未知模块名' ;;
esac
evidence_root="${SCBS_EVIDENCE_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
evidence_gpu="${SCBS_EVIDENCE_GPU:-0}"; evidence_port="${SCBS_EVIDENCE_PORT:-7781}"
evidence_conda="${SCBS_EVIDENCE_CONDA:-${CONDA_EXE:-conda}}"; evidence_env="${SCBS_EVIDENCE_ENV:-wq}"
evidence_helper='tools/diagnostics/scbs_instance_evidence_experiment.py'
[[ "$evidence_root" == /* ]] || evidence_fail '输出根目录必须为绝对路径'
[[ "$evidence_gpu" =~ ^[0-9]+$ && "$evidence_port" =~ ^[0-9]+$ ]] || evidence_fail 'GPU/端口必须为整数'
[[ "$evidence_port" -ge 1024 && "$evidence_port" -le 65535 ]] || evidence_fail '端口范围1024..65535'
for evidence_file in train.py "$evidence_helper" tools/diagnostics/test_instance_evidence.py \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  [[ -f "$evidence_file" ]] || evidence_fail "缺少文件：$evidence_file"
done
# Check ALL outputs before any training; preserve every old run and partial run.
for evidence_mode in "${evidence_modes[@]}"; do
  evidence_config="configs/dfine/dfine_s_scbs_hrw_110_AQS_${evidence_mode}.yml"
  evidence_out="$evidence_root/dfine_s_scbs_hrw_epochs110_AQS_${evidence_mode}_seed0_${evidence_run}"
  [[ -f "$evidence_config" ]] || evidence_fail "缺少配置：$evidence_config"
  [[ ! -e "$evidence_out" && ! -L "$evidence_out" ]] || evidence_fail "保留旧结果，未启动任何训练：$evidence_out"
  printf '计划：AQS-L2 R1 + %s\n配置：%s\n输出：%s\n' "$evidence_mode" "$evidence_config" "$evidence_out"
done
if [[ "$evidence_dry" == 1 ]]; then
  printf '仅演练：110/100轮，batch32，seed0，从头训练，顺序运行；不创建输出、不连接服务器。\n'
  exit 0
fi
command -v "$evidence_conda" >/dev/null 2>&1 || evidence_fail '找不到Conda'
evidence_aqs_default="$evidence_root/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval"
evidence_run2_default="$evidence_root/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval"
evidence_aqs_reference="${SCBS_EVIDENCE_AQS_REFERENCE:-$evidence_aqs_default}"
evidence_run2_reference="${SCBS_EVIDENCE_RUN2_REFERENCE:-$evidence_run2_default}"
if [[ -n "${SCBS_EVIDENCE_AQS_REFERENCE:-}" ]]; then
  [[ -f "$evidence_aqs_reference/analysis/summary.json" && -f "$evidence_aqs_reference/analysis/gt_records.jsonl" && -f "$evidence_aqs_reference/evaluation/query_diagnostics/metadata.json" ]] || evidence_fail '指定的AQS诊断目录不完整'
fi
if [[ -n "${SCBS_EVIDENCE_RUN2_REFERENCE:-}" ]]; then
  [[ -f "$evidence_run2_reference/analysis/summary.json" && -f "$evidence_run2_reference/analysis/gt_records.jsonl" && -f "$evidence_run2_reference/evaluation/query_diagnostics/metadata.json" ]] || evidence_fail '指定的Run2诊断目录不完整'
fi
export CUDA_VISIBLE_DEVICES="$evidence_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
evidence_check_gpu_port() {
  "$evidence_conda" run --no-capture-output -n "$evidence_env" python - "$evidence_gpu" "$evidence_port" <<'PY'
import socket, subprocess, sys
from pathlib import Path
processes = subprocess.run(['nvidia-smi','-i',sys.argv[1],'--query-compute-apps=pid,process_name',
    '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
desktop = {'Xorg','gnome-shell','gnome-remote-desktop-daemon'}
busy = [line for line in processes.splitlines() if Path(line.split(',',1)[-1].strip()).name not in desktop]
assert not busy, f'GPU仍有计算进程；不终止其他项目：{busy}'
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s: s.bind(('0.0.0.0',int(sys.argv[2])))
print('GPU/端口空闲检查通过。',flush=True)
PY
}
evidence_check_gpu_port
"$evidence_conda" run --no-capture-output -n "$evidence_env" python -u "$evidence_helper" preflight "${evidence_modes[@]}"
printf '\n合成正确性测试（包含服务器CUDA FP16和batch32局部分支；不是效果验证）。\n'
"$evidence_conda" run --no-capture-output -n "$evidence_env" python -u tools/diagnostics/test_instance_evidence.py
for evidence_mode in "${evidence_modes[@]}"; do
  evidence_check_gpu_port
  evidence_config="configs/dfine/dfine_s_scbs_hrw_110_AQS_${evidence_mode}.yml"
  evidence_out="$evidence_root/dfine_s_scbs_hrw_epochs110_AQS_${evidence_mode}_seed0_${evidence_run}"
  mkdir -p -- "$evidence_root"
  mkdir -- "$evidence_out"
  trap 'evidence_status=$?; printf "实验未完成，退出码=%s；保留输出：%s\n" "$evidence_status" "$evidence_out" >&2; exit "$evidence_status"' ERR
  "$evidence_conda" run --no-capture-output -n "$evidence_env" python -u "$evidence_helper" snapshot "$evidence_mode" "$evidence_out"
  printf '\n开始从头训练：AQS + %s；日志同时在页面显示和保存。\n' "$evidence_mode"
  "$evidence_conda" run --no-capture-output -n "$evidence_env" \
  torchrun --master_port="$evidence_port" --nproc_per_node=1 train.py \
  -c "$evidence_config" --output-dir "$evidence_out" --use-amp --seed=0 \
  2>&1 | tee "$evidence_out/console.log"
  test -f "$evidence_out/best_stg2.pth"
  "$evidence_conda" run --no-capture-output -n "$evidence_env" \
  python -u tools/diagnostics/export_aqs_parameters.py --checkpoint "$evidence_out/best_stg2.pth" --temperature .10
  "$evidence_conda" run --no-capture-output -n "$evidence_env" \
  python -u "$evidence_helper" parameters "$evidence_out/best_stg2.pth" "$evidence_out/best_stg2_${evidence_mode}_parameters.json"
  printf '\n导出最佳EMA：原COCO评估、混淆矩阵、全部query诊断。\n'
  "$evidence_conda" run --no-capture-output -n "$evidence_env" \
  torchrun --master_port="$evidence_port" --nproc_per_node=1 train.py \
  -c "$evidence_config" -r "$evidence_out/best_stg2.pth" --test-only \
  --output-dir "$evidence_out/diagnostics_eval/evaluation" --use-amp --seed=0 \
  -u export_confusion_matrix=true export_query_diagnostics=true \
  query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
  2>&1 | tee "$evidence_out/diagnostics_console.log"
  "$evidence_conda" run --no-capture-output -n "$evidence_env" \
  python -u tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$evidence_out/diagnostics_eval/evaluation" --output-dir "$evidence_out/diagnostics_eval/analysis" \
  2>&1 | tee "$evidence_out/analysis_console.log"
  test -s "$evidence_out/diagnostics_eval/analysis/coco_per_class_and_size.json"
  test -s "$evidence_out/diagnostics_eval/analysis/best_query_cross_classification.json"
  for evidence_label in AQS_L2_R1 BaselineRun2; do
    evidence_reference="$evidence_aqs_reference"
    [[ "$evidence_label" != BaselineRun2 ]] || evidence_reference="$evidence_run2_reference"
    if [[ -f "$evidence_reference/analysis/summary.json" && -f "$evidence_reference/analysis/gt_records.jsonl" && -f "$evidence_reference/evaluation/query_diagnostics/metadata.json" ]]; then
      "$evidence_conda" run --no-capture-output -n "$evidence_env" \
      python -u "$evidence_helper" compare "$evidence_out" "$evidence_reference" "$evidence_label"
    else
      printf '未发现%s完整诊断参考：%s；本组结果已导出，可后续离线比较。\n' "$evidence_label" "$evidence_reference"
    fi
  done
  printf '\n本组完成：%s\n' "$evidence_out"
done
printf '\n所选实验全部完成；优先与AQS单独版比较增量，再与Baseline Run2比较。\n'
