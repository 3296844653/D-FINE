#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
scf_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
scf_dry=0
if [[ "${1:-}" == --dry-run ]]; then scf_dry=1; shift; fi
[[ "$#" -le 1 ]] || scf_fail '用法：[--dry-run] [run1|run2...]'
scf_run="${1:-run1}"
[[ "$scf_run" =~ ^run[1-9][0-9]*$ ]] || scf_fail '编号须为run1/run2等'
scf_root="${SCBS_SCF_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
scf_gpu="${SCBS_SCF_GPU:-0}"; scf_port="${SCBS_SCF_PORT:-7781}"
scf_conda="${SCBS_SCF_CONDA:-${CONDA_EXE:-conda}}"; scf_env="${SCBS_SCF_ENV:-wq}"
scf_config='configs/dfine/dfine_s_scbs_hrw_110_AQS_SubpixelContextFusion.yml'
scf_helper='tools/diagnostics/scbs_subpixel_context_experiment.py'
scf_out="$scf_root/dfine_s_scbs_hrw_epochs110_AQS_SubpixelContextFusion_seed0_${scf_run}"
[[ "$scf_root" == /* ]] || scf_fail '输出根目录须为绝对路径'
[[ "$scf_gpu" =~ ^[0-9]+$ && "$scf_port" =~ ^[0-9]+$ ]] || scf_fail 'GPU/端口须为整数'
[[ "$scf_port" -ge 1024 && "$scf_port" -le 65535 ]] || scf_fail '端口须为1024..65535'
[[ ! -e "$scf_out" && ! -L "$scf_out" ]] || scf_fail "保留旧结果，未启动：$scf_out"
for scf_file in train.py "$scf_config" "$scf_helper" src/zoo/dfine/subpixel_context_fusion.py \
    tools/diagnostics/test_subpixel_context_fusion.py tools/diagnostics/eval_subpixel_context.py \
    tools/diagnostics/export_aqs_parameters.py tools/diagnostics/analyze_detection_diagnostics.py; do
  [[ -f "$scf_file" ]] || scf_fail "缺少文件：$scf_file"
done
printf 'AQS-L2 R1 + SubpixelContextFusion；110/100轮，batch32，seed0，原损失。\n输出：%s\n' "$scf_out"
if [[ "$scf_dry" == 1 ]]; then
  printf '仅演练；不创建目录、不连接服务器、不启动训练。\n'; exit 0
fi
export CUDA_VISIBLE_DEVICES="$scf_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
scf_check_gpu_port() {
"$scf_conda" run --no-capture-output -n "$scf_env" python - "$scf_gpu" "$scf_port" <<'PY'
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
scf_check_gpu_port
"$scf_conda" run --no-capture-output -n "$scf_env" python -u "$scf_helper" preflight
printf '\n合成正确性测试（包括CUDA）；不是效果验证。\n'
SCF_REQUIRE_CUDA_TEST=1 "$scf_conda" run --no-capture-output -n "$scf_env" \
python -u tools/diagnostics/test_subpixel_context_fusion.py
scf_check_gpu_port
mkdir -p -- "$scf_root"
mkdir -- "$scf_out"
trap 'scf_status=$?; printf "未完成，退出码=%s；输出已保留：%s\n" "$scf_status" "$scf_out" >&2; exit "$scf_status"' ERR
"$scf_conda" run --no-capture-output -n "$scf_env" python -u "$scf_helper" snapshot "$scf_out"
printf '\n开始按相同配方从头训练；AQS继续学习，不冻结、不加载旧实验权重。\n'
"$scf_conda" run --no-capture-output -n "$scf_env" \
torchrun --master_port="$scf_port" --nproc_per_node=1 train.py \
-c "$scf_config" --output-dir "$scf_out" --use-amp --seed=0 2>&1 | tee "$scf_out/console.log"
test -s "$scf_out/best_stg2.pth"
"$scf_conda" run --no-capture-output -n "$scf_env" python -u tools/diagnostics/export_aqs_parameters.py \
--checkpoint "$scf_out/best_stg2.pth" --temperature .10
"$scf_conda" run --no-capture-output -n "$scf_env" python -u "$scf_helper" parameters \
"$scf_out/best_stg2.pth" "$scf_out/best_stg2_SubpixelContextFusion_parameters.json"
for scf_mode in on off; do
  scf_diag="$scf_out/diagnostics_eval"; scf_flag=true
  if [[ "$scf_mode" == off ]]; then scf_diag="$scf_out/SubpixelContextFusion_off_eval"; scf_flag=false; fi
  "$scf_conda" run --no-capture-output -n "$scf_env" \
  torchrun --master_port="$scf_port" --nproc_per_node=1 tools/diagnostics/eval_subpixel_context.py \
  --module-mode "$scf_mode" -c "$scf_config" -r "$scf_out/best_stg2.pth" --test-only \
  --output-dir "$scf_diag/evaluation" --use-amp --seed=0 \
  -u export_confusion_matrix=true export_query_diagnostics=true \
  query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
  diagnostic_SubpixelContextFusion_output_enabled="$scf_flag" 2>&1 | tee "$scf_out/${scf_mode}_eval_console.log"
  "$scf_conda" run --no-capture-output -n "$scf_env" python -u tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$scf_diag/evaluation" --output-dir "$scf_diag/analysis" \
  2>&1 | tee "$scf_out/${scf_mode}_analysis_console.log"
  test -s "$scf_diag/analysis/summary.json"
done
"$scf_conda" run --no-capture-output -n "$scf_env" python -u "$scf_helper" on-off "$scf_out"
scf_compare_failed=0
for scf_label in AQS_L2_R1 BaselineRun2; do
  scf_reference="${SCBS_SCF_AQS_REFERENCE:-$scf_root/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval}"
  if [[ "$scf_label" == AQS_L2_R1 && -z "${SCBS_SCF_AQS_REFERENCE:-}" && -f "${scf_reference}_v2/analysis/summary.json" ]]; then
    scf_reference="${scf_reference}_v2"
  fi
  if [[ "$scf_label" == BaselineRun2 ]]; then
    scf_reference="${SCBS_SCF_RUN2_REFERENCE:-$scf_root/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_export1}"
    if [[ -z "${SCBS_SCF_RUN2_REFERENCE:-}" && ! -f "$scf_reference/analysis/summary.json" ]]; then
      scf_reference="$scf_root/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval"
    fi
  fi
  if [[ -f "$scf_reference/analysis/summary.json" && -f "$scf_reference/analysis/gt_records.jsonl" && -f "$scf_reference/evaluation/query_diagnostics/metadata.json" ]]; then
    if ! "$scf_conda" run --no-capture-output -n "$scf_env" python -u "$scf_helper" compare "$scf_out" "$scf_reference" "$scf_label"; then
      scf_compare_failed=1
      printf '%s参考不兼容；保留独立诊断并继续打包，不伪造差值。\n' "$scf_label" >&2
    fi
  else
    printf '未找到%s完整诊断参考，不伪造比较：%s\n' "$scf_label" "$scf_reference"
  fi
done
"$scf_conda" run --no-capture-output -n "$scf_env" python -u "$scf_helper" pack "$scf_out"
if [[ "$scf_compare_failed" == 1 ]]; then
  printf '训练/ON-OFF/压缩包已完成；参考比较未完成，请核对参考诊断路径。\n' >&2
  exit 2
fi
printf '\n训练、诊断及打包完成：%s\n' "$scf_out"
