#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
der_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
der_dry=0
if [[ "${1:-}" == --dry-run ]]; then der_dry=1; shift; fi
[[ "$#" -le 1 ]] || der_fail '用法：[--dry-run] [run1|run2...]'
der_run="${1:-run1}"
[[ "$der_run" =~ ^run[1-9][0-9]*$ ]] || der_fail '运行编号应为run1/run2等'
der_root="${SCBS_DER_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
der_gpu="${SCBS_DER_GPU:-0}"; der_port="${SCBS_DER_PORT:-7781}"
der_conda="${SCBS_DER_CONDA:-${CONDA_EXE:-conda}}"; der_env="${SCBS_DER_ENV:-wq}"
der_config='configs/dfine/dfine_s_scbs_hrw_110_AQS_DiscriminativeEvidenceL2.yml'
der_helper='tools/diagnostics/scbs_discriminative_evidence_experiment.py'
der_out="$der_root/dfine_s_scbs_hrw_epochs110_AQS_DiscriminativeEvidenceL2_seed0_${der_run}"
[[ "$der_root" == /* ]] || der_fail '输出根目录须为绝对路径'
[[ "$der_gpu" =~ ^[0-9]+$ && "$der_port" =~ ^[0-9]+$ ]] || der_fail 'GPU/端口须为整数'
[[ "$der_port" -ge 1024 && "$der_port" -le 65535 ]] || der_fail '端口须为1024..65535'
[[ ! -e "$der_out" && ! -L "$der_out" ]] || der_fail "保留旧结果，未启动：$der_out"
for der_file in train.py "$der_config" "$der_helper" tools/diagnostics/test_discriminative_evidence.py \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  [[ -f "$der_file" ]] || der_fail "缺少文件：$der_file"
done
printf 'AQS-L2 R1 + DiscriminativeEvidenceL2；110/100轮，batch32，seed0，原损失。\n输出：%s\n' "$der_out"
if [[ "$der_dry" == 1 ]]; then
  printf '仅演练；不创建目录、不连接服务器、不启动训练。\n'; exit 0
fi
export CUDA_VISIBLE_DEVICES="$der_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
der_check_gpu_port() {
"$der_conda" run --no-capture-output -n "$der_env" python - "$der_gpu" "$der_port" <<'PY'
import socket, subprocess, sys
from pathlib import Path
rows = subprocess.run(['nvidia-smi','-i',sys.argv[1],'--query-compute-apps=pid,process_name',
    '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
desktop = {'Xorg','gnome-shell','gnome-remote-desktop-daemon'}
busy = [r for r in rows.splitlines() if Path(r.split(',',1)[-1].strip()).name not in desktop]
assert not busy, f'GPU被占用；不停止其他项目：{busy}'
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s: s.bind(('0.0.0.0',int(sys.argv[2])))
print('GPU/端口检查通过。',flush=True)
PY
}
der_check_gpu_port
"$der_conda" run --no-capture-output -n "$der_env" python -u "$der_helper" preflight
printf '\n合成正确性测试；服务器执行CUDA测试，不是效果验证。\n'
"$der_conda" run --no-capture-output -n "$der_env" python -u tools/diagnostics/test_discriminative_evidence.py
der_check_gpu_port
mkdir -p -- "$der_root"
mkdir -- "$der_out"
trap 'der_status=$?; printf "未完成，退出码=%s；已保留输出：%s\n" "$der_status" "$der_out" >&2; exit "$der_status"' ERR
"$der_conda" run --no-capture-output -n "$der_env" python -u "$der_helper" snapshot "$der_out"
printf '\n开始从头训练；终端实时显示日志。不冻结AQS，不读取旧训练权重。\n'
"$der_conda" run --no-capture-output -n "$der_env" \
torchrun --master_port="$der_port" --nproc_per_node=1 train.py \
-c "$der_config" --output-dir "$der_out" --use-amp --seed=0 2>&1 | tee "$der_out/console.log"
test -s "$der_out/best_stg2.pth"
"$der_conda" run --no-capture-output -n "$der_env" python -u tools/diagnostics/export_aqs_parameters.py \
--checkpoint "$der_out/best_stg2.pth" --temperature .10
"$der_conda" run --no-capture-output -n "$der_env" python -u "$der_helper" parameters \
"$der_out/best_stg2.pth" "$der_out/best_stg2_DiscriminativeEvidenceL2_parameters.json"

# SAME trained EMA, only the new query residual is toggled. No checkpoint/source
# changes; AQS stays ON. OFF is NOT a separately trained AQS baseline.
der_wrapper='import os,runpy
if os.environ["DER_EVAL_MODE"] == "off":
    from src.zoo.dfine.discriminative_evidence import DiscriminativeEvidenceL2
    original = DiscriminativeEvidenceL2.forward
    def disabled(self,*args,**kwargs):
        self.enabled = False
        return original(self,*args,**kwargs)
    DiscriminativeEvidenceL2.forward = disabled
print("DiscriminativeEvidenceL2 runtime:",os.environ["DER_EVAL_MODE"],flush=True)
runpy.run_path("train.py",run_name="__main__")'
for der_mode in on off; do
  der_diag="$der_out/diagnostics_eval"
  der_flag=true
  if [[ "$der_mode" == off ]]; then der_diag="$der_out/DiscriminativeEvidenceL2_off_eval"; der_flag=false; fi
  printf '\n最佳EMA诊断，新模块=%s，AQS仍开启。\n' "$der_mode"
  DER_EVAL_MODE="$der_mode" "$der_conda" run --no-capture-output -n "$der_env" \
  torchrun --master_port="$der_port" --nproc_per_node=1 --no-python python -u -c "$der_wrapper" \
  -c "$der_config" -r "$der_out/best_stg2.pth" --test-only \
  --output-dir "$der_diag/evaluation" --use-amp --seed=0 \
  -u export_confusion_matrix=true export_query_diagnostics=true \
  query_diagnostic_conf_thresh=0.5 query_diagnostic_iou_thresh=0.5 query_diagnostic_neighbor_iou_thresh=0.8 \
  diagnostic_DiscriminativeEvidenceL2_output_enabled="$der_flag" \
  2>&1 | tee "$der_out/${der_mode}_diagnostics_console.log"
  "$der_conda" run --no-capture-output -n "$der_env" python -u tools/diagnostics/analyze_detection_diagnostics.py \
  --input-dir "$der_diag/evaluation" --output-dir "$der_diag/analysis" \
  2>&1 | tee "$der_out/${der_mode}_analysis_console.log"
  test -s "$der_diag/analysis/summary.json"
done
"$der_conda" run --no-capture-output -n "$der_env" python -u "$der_helper" on-off "$der_out"
for der_label in AQS_L2_R1 BaselineRun2; do
  der_reference="${SCBS_DER_AQS_REFERENCE:-$der_root/dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1/diagnostics_eval}"
  if [[ "$der_label" == BaselineRun2 ]]; then
    der_reference="${SCBS_DER_RUN2_REFERENCE:-$der_root/dfine_s_scbs_hrw_epochs110_seed0_run2/diagnostics_eval}"
  fi
  if [[ -f "$der_reference/analysis/summary.json" && -f "$der_reference/analysis/gt_records.jsonl" && -f "$der_reference/evaluation/query_diagnostics/metadata.json" ]]; then
    "$der_conda" run --no-capture-output -n "$der_env" python -u "$der_helper" compare "$der_out" "$der_reference" "$der_label"
  else
    printf '未找到%s完整参考；新结果已保存，未伪造比较：%s\n' "$der_label" "$der_reference"
  fi
done
printf '\n训练、最佳EMA诊断及ON/OFF评估完成：%s\n' "$der_out"
