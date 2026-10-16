#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sca_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
sca_dry=0
if [[ "${1:-}" == --dry-run ]]; then sca_dry=1; shift; fi
[[ "$#" -le 1 ]] || sca_fail '用法：[--dry-run] [run1|run2...]'
sca_run="${1:-run1}"
[[ "$sca_run" =~ ^run[1-9][0-9]*$ ]] || sca_fail '编号须为run1/run2等'
sca_root="${SCBS_SCA_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
sca_gpu="${SCBS_SCA_GPU:-0}"; sca_port="${SCBS_SCA_PORT:-7781}"
sca_conda="${SCBS_SCA_CONDA:-${CONDA_EXE:-conda}}"; sca_env="${SCBS_SCA_ENV:-wq}"
sca_config='configs/dfine/dfine_s_scbs_hrw_110_AQS_SCA.yml'
sca_helper='tools/diagnostics/scbs_sca_experiment.py'
sca_out="$sca_root/dfine_s_scbs_hrw_epochs110_AQS_SCA_seed0_${sca_run}"
[[ "$sca_root" == /* ]] || sca_fail '输出根目录须为绝对路径'
[[ "$sca_gpu" =~ ^[0-9]+$ && "$sca_port" =~ ^[0-9]+$ ]] || sca_fail 'GPU/端口须为整数'
[[ "$sca_port" -ge 1024 && "$sca_port" -le 65535 ]] || sca_fail '端口须为1024..65535'
[[ ! -e "$sca_out" && ! -L "$sca_out" ]] || sca_fail "保留旧结果，未启动：$sca_out"
for sca_file in train.py "$sca_config" "$sca_helper" tools/diagnostics/test_scbs_aqs_sca.py \
    tools/diagnostics/export_aqs_parameters.py tools/diagnostics/analyze_detection_diagnostics.py; do
  [[ -f "$sca_file" ]] || sca_fail "缺少文件：$sca_file"
done
printf 'AQS-L2 R1 + SCA；仅修改三项增强参数，110/100轮，batch32，seed0，原结构/损失。\n输出：%s\n' "$sca_out"
if [[ "$sca_dry" == 1 ]]; then
  printf '仅演练；不创建目录、不连接服务器、不启动训练。\n'; exit 0
fi
export CUDA_VISIBLE_DEVICES="$sca_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
sca_check_gpu_port() {
"$sca_conda" run --no-capture-output -n "$sca_env" python - "$sca_gpu" "$sca_port" <<'PY'
import socket,subprocess,sys
from pathlib import Path
rows=subprocess.run(['nvidia-smi','-i',sys.argv[1],'--query-compute-apps=pid,process_name',
    '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
desktop={'Xorg','gnome-shell','gnome-remote-desktop-daemon'}
busy=[r for r in rows.splitlines() if Path(r.split(',',1)[-1].strip()).name not in desktop]
assert not busy,f'GPU被占用；不停止其他项目：{busy}'
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s: s.bind(('0.0.0.0',int(sys.argv[2])))
print('GPU/端口检查通过。',flush=True)
PY
}
sca_check_gpu_port
"$sca_conda" run --no-capture-output -n "$sca_env" python -u "$sca_helper" preflight
"$sca_conda" run --no-capture-output -n "$sca_env" python -u tools/diagnostics/test_scbs_aqs_sca.py
sca_check_gpu_port
mkdir -p -- "$sca_root"
mkdir -- "$sca_out"
trap 'sca_status=$?; printf "未完成，退出码=%s；输出已保留：%s\n" "$sca_status" "$sca_out" >&2; exit "$sca_status"' ERR
"$sca_conda" run --no-capture-output -n "$sca_env" python -u "$sca_helper" snapshot "$sca_out"
"$sca_conda" run --no-capture-output -n "$sca_env" \
torchrun --master_port="$sca_port" --nproc_per_node=1 train.py \
-c "$sca_config" --output-dir "$sca_out" --use-amp --seed=0 2>&1 | tee "$sca_out/console.log"
test -s "$sca_out/best_stg2.pth"
"$sca_conda" run --no-capture-output -n "$sca_env" python -u tools/diagnostics/export_aqs_parameters.py \
--checkpoint "$sca_out/best_stg2.pth" --temperature .10
sca_diag="$sca_out/diagnostics_eval"
"$sca_conda" run --no-capture-output -n "$sca_env" \
torchrun --master_port="$sca_port" --nproc_per_node=1 train.py \
-c "$sca_config" -r "$sca_out/best_stg2.pth" --test-only \
--output-dir "$sca_diag/evaluation" --use-amp --seed=0 \
-u export_confusion_matrix=true export_query_diagnostics=true \
query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
2>&1 | tee "$sca_out/diagnostics_console.log"
"$sca_conda" run --no-capture-output -n "$sca_env" python -u tools/diagnostics/analyze_detection_diagnostics.py \
--input-dir "$sca_diag/evaluation" --output-dir "$sca_diag/analysis" \
2>&1 | tee "$sca_out/analysis_console.log"
test -s "$sca_diag/analysis/summary.json"
sca_compare_failed=0
for sca_label in AQS_L2_R1 BaselineRun2; do
  sca_reference="${SCBS_SCA_AQS_REFERENCE:-$sca_root/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval}"
  if [[ "$sca_label" == AQS_L2_R1 && -z "${SCBS_SCA_AQS_REFERENCE:-}" && -f "${sca_reference}_v2/analysis/summary.json" ]]; then
    sca_reference="${sca_reference}_v2"
  fi
  if [[ "$sca_label" == BaselineRun2 ]]; then
    sca_reference="${SCBS_SCA_RUN2_REFERENCE:-$sca_root/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_export1}"
    if [[ -z "${SCBS_SCA_RUN2_REFERENCE:-}" && ! -f "$sca_reference/analysis/summary.json" ]]; then
      sca_reference="$sca_root/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval"
    fi
  fi
  if [[ -f "$sca_reference/analysis/summary.json" && -f "$sca_reference/analysis/gt_records.jsonl" && -f "$sca_reference/evaluation/query_diagnostics/metadata.json" ]]; then
    if ! "$sca_conda" run --no-capture-output -n "$sca_env" python -u "$sca_helper" compare "$sca_out" "$sca_reference" "$sca_label"; then
      sca_compare_failed=1
      printf '%s参考不兼容；保留独立诊断并继续打包，不伪造比较。\n' "$sca_label" >&2
    fi
  else
    printf '未找到%s完整参考，不伪造比较：%s\n' "$sca_label" "$sca_reference"
  fi
done
"$sca_conda" run --no-capture-output -n "$sca_env" python -u "$sca_helper" pack "$sca_out"
if [[ "$sca_compare_failed" == 1 ]]; then
  printf '训练/诊断/压缩包已完成；参考比较未完成，请核对参考路径。\n' >&2; exit 2
fi
printf '\n训练及诊断完成：%s\n' "$sca_out"
