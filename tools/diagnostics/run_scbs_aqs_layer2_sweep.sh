#!/usr/bin/env bash
# Independent, sequential 110-epoch runs. No resume, tuning, deletion or overwrites.
set -Eeuo pipefail

aqs_fail() { printf '错误：%s\n' "$*" >&2; exit 1; }
aqs_usage() {
  printf '%s\n' \
    '用法：bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh [--dry-run] [组别 ...]' \
    '默认顺序：R1 R2 R3 TAU1 TAU2 T1 T2（不重复已有原L2对照）' \
    '可选CONTROL：另跑一次原L2参数对照，使用独立control目录。' \
    '示例：bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh R1 R2 R3' \
    '示例：bash tools/diagnostics/run_scbs_aqs_layer2_sweep.sh --dry-run' \
    '环境变量：AQS_SWEEP_OUTPUT_ROOT、AQS_SWEEP_RUN_ID（默认1）、' \
    '          AQS_SWEEP_GPU（默认0）、AQS_SWEEP_CONDA_ENV（默认wq）、AQS_SWEEP_CONDA（默认conda）。' \
    '已存在的输出目录会停止整个队列；不自动跳过、恢复或覆盖。'
}

aqs_groups=()
aqs_dry_run=0
for aqs_arg in "$@"; do
  case "$aqs_arg" in
    --dry-run) aqs_dry_run=1 ;;
    --help|-h) aqs_usage; exit 0 ;;
    --*) aqs_fail "未知选项：$aqs_arg" ;;
    *) aqs_groups+=("$aqs_arg") ;;
  esac
done
if [[ ${#aqs_groups[@]} -eq 0 ]]; then
  aqs_groups=(R1 R2 R3 TAU1 TAU2 T1 T2)
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
aqs_output_root="${AQS_SWEEP_OUTPUT_ROOT:-/media/a5/5号机移动盘2/wq_workspace/wq_s_scbs_hrw}"
aqs_run_id="${AQS_SWEEP_RUN_ID:-1}"
aqs_gpu="${AQS_SWEEP_GPU:-0}"
aqs_env="${AQS_SWEEP_CONDA_ENV:-wq}"
# Prefer the executable, not a shell-only conda function used by conda init.
aqs_conda="${AQS_SWEEP_CONDA:-${CONDA_EXE:-conda}}"
[[ "$aqs_output_root" == /* ]] || aqs_fail '输出根目录必须是绝对路径'
[[ "$aqs_run_id" =~ ^[1-9][0-9]*$ ]] || aqs_fail 'RUN_ID必须是正整数'
[[ "$aqs_gpu" =~ ^[0-9]+$ ]] || aqs_fail 'GPU必须是单个非负整数编号'
command -v "$aqs_conda" >/dev/null 2>&1 || aqs_fail "找不到Conda：$aqs_conda"
test -f train.py || aqs_fail '缺少项目根目录的train.py'

aqs_configs=()
aqs_outputs=()
aqs_specs=()
aqs_seen=' '
for aqs_group in "${aqs_groups[@]}"; do
  case "$aqs_seen" in *" $aqs_group "*) aqs_fail "重复组别：$aqs_group" ;; esac
  aqs_seen+="$aqs_group "
  aqs_tau=0.50; aqs_rho=0.05; aqs_temp=0.10
  case "$aqs_group" in
    R1) aqs_tag=r010; aqs_rho=0.10 ;;
    R2) aqs_tag=r020; aqs_rho=0.20 ;;
    R3) aqs_tag=r025; aqs_rho=0.25 ;;
    TAU1) aqs_tag=tau045; aqs_tau=0.45 ;;
    TAU2) aqs_tag=tau055; aqs_tau=0.55 ;;
    T1) aqs_tag=temp005; aqs_temp=0.05 ;;
    T2) aqs_tag=temp020; aqs_temp=0.20 ;;
    CONTROL) aqs_tag=control ;;
    *) aqs_fail "未知组别：${aqs_group}；请使用R1/R2/R3/TAU1/TAU2/T1/T2/CONTROL" ;;
  esac
  if [[ "$aqs_group" == CONTROL ]]; then
    aqs_config='configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2.yml'
  else
    aqs_config="configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_${aqs_tag}.yml"
  fi
  aqs_out="$aqs_output_root/dfine_s_scbs_hrw_epochs110_aqs_layer2_${aqs_tag}_seed0_run${aqs_run_id}"
  test -f "$aqs_config" || aqs_fail "缺少配置：$aqs_config"
  [[ ! -e "$aqs_out" ]] || aqs_fail "输出目录已存在，未启动任何训练：${aqs_out}；请排除此组或使用新RUN_ID"
  aqs_configs+=("$aqs_config")
  aqs_outputs+=("$aqs_out")
  aqs_specs+=("$aqs_group|$aqs_config|$aqs_tau|$aqs_rho|$aqs_temp")
  printf '%s：threshold=%s residual_init=%s temperature=%s\n  输出：%s\n' \
    "$aqs_group" "$aqs_tau" "$aqs_rho" "$aqs_temp" "$aqs_out"
done

# Check ALL selected configurations before launching anything. Dry-run intentionally
# skips server-only data/GPU existence checks and never creates output directories.
PYTHONUNBUFFERED=1 MPLBACKEND=Agg "$aqs_conda" run --no-capture-output -n "$aqs_env" \
python - "$aqs_dry_run" "${aqs_specs[@]}" <<'PY'
import copy
import sys
from pathlib import Path
from src.core.yaml_utils import load_config

dry_run = sys.argv[1] == '1'
reference = load_config('configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2.yml', cfg={})
assert reference['DFINETransformer']['use_aqs_refine'] is True
assert reference['DFINETransformer']['aqs_apply_layer'] == 2
assert reference['epochs'] == 110 and reference['num_classes'] == 3
assert reference['train_dataloader']['total_batch_size'] == 32
assert reference['train_dataloader']['dataset']['transforms']['policy']['epoch'] == 100
assert reference['train_dataloader']['collate_fn']['stop_epoch'] == 100
assert tuple(float(reference['DFINETransformer'][k]) for k in
             ('aqs_threshold', 'aqs_residual_init', 'aqs_temperature')) == (.5, .05, .1)
for spec in sys.argv[2:]:
    group, path, tau, rho, temp = spec.split('|')
    actual = load_config(path, cfg={})
    expected = copy.deepcopy(reference)
    for cfg in (expected, actual):
        cfg.pop('__include__', None)
        cfg.pop('output_dir', None)
    for name, value in zip(('aqs_threshold', 'aqs_residual_init', 'aqs_temperature'),
                           (tau, rho, temp)):
        expected['DFINETransformer'][name] = float(value)
    assert actual == expected, f'{group}: 参数以外的训练设置发生变化，停止'
    assert actual['DFINETransformer'].get('aqs_apply_layers') is None, f'{group}: 不得使用双层AQS'
    if not dry_run:
        for split in ('train', 'val'):
            data = actual[f'{split}_dataloader']['dataset']
            assert Path(data['ann_file']).is_file(), f"缺少标注：{data['ann_file']}"
            assert Path(data['img_folder']).is_dir(), f"缺少图片目录：{data['img_folder']}"
    print(f'{group}: 配置检查通过，仅L2，110/100，batch32，原数据和损失不变', flush=True)
PY

aqs_make_command() {
  aqs_cmd=(env "CUDA_VISIBLE_DEVICES=$aqs_gpu" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    "$aqs_conda" run --no-capture-output -n "$aqs_env"
    torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=1
    train.py -c "${aqs_configs[$aqs_index]}" --output-dir "${aqs_outputs[$aqs_index]}"
    --use-amp --seed=0)
}
if [[ "$aqs_dry_run" -eq 1 ]]; then
  for ((aqs_index=0; aqs_index<${#aqs_groups[@]}; aqs_index++)); do
    aqs_make_command
    # Start a C-locale shell: an inline locale assignment does not fix Bash 3's
    # internal multibyte %q state when the inherited C.UTF-8 locale is unsupported.
    printf '命令：'
    LC_ALL=C "$BASH" -c 'printf "%q " "$@"' _ "${aqs_cmd[@]}"
    printf '\n'
  done
  printf '演练完成：未启动训练，未创建输出目录；尚未检查服务器数据/GPU。\n'
  exit 0
fi

# This lock only prevents a second copy of this sweep from using the same GPU.
# It does not kill jobs or claim exclusive access against other users' programs.
if command -v flock >/dev/null 2>&1; then
  exec 9>"/tmp/dfine-aqs-layer2-sweep-gpu${aqs_gpu}.lock"
  flock -n 9 || aqs_fail "同GPU的另一份AQS队列仍在运行：GPU$aqs_gpu"
fi
aqs_check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '提示：未找到nvidia-smi，无法检查其他GPU任务；训练进程将验证CUDA。\n' >&2
    return
  fi
  nvidia-smi -i "$aqs_gpu"
  aqs_gpu_pids="$(nvidia-smi -i "$aqs_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
  while IFS= read -r aqs_pid; do
    aqs_pid="${aqs_pid//[[:space:]]/}"
    [[ "$aqs_pid" =~ ^[0-9]+$ ]] || continue
    # Do not block on a disappeared PID cached briefly by nvidia-smi.
    if aqs_process="$(ps -p "$aqs_pid" -o comm= 2>&1)"; then
      :
    elif [[ -z "$aqs_process" ]]; then
      continue  # PID has exited; ps returned no process and no error text.
    else
      aqs_fail "无法读取GPU进程PID=${aqs_pid}的状态，不能确认GPU空闲；请人工检查"
    fi
    [[ -n "$aqs_process" ]] || continue
    case "$aqs_process" in
      # Linux comm may truncate gnome-remote-desktop to 15 characters.
      *gnome-remote-de*|*gnome-shell*|*/Xorg|Xorg) continue ;;
    esac
    aqs_fail "GPU${aqs_gpu}有其他计算进程PID=${aqs_pid}；没有停止该进程，请等待GPU空闲"
  done <<< "$aqs_gpu_pids"
}

aqs_current_group=preflight
aqs_current_output='尚未创建'
trap 'aqs_status=$?; printf "队列停止：组别=%s，退出码=%s；保留已有结果：%s\n" "$aqs_current_group" "$aqs_status" "$aqs_current_output" >&2; exit "$aqs_status"' ERR
mkdir -p "$aqs_output_root"
for ((aqs_index=0; aqs_index<${#aqs_groups[@]}; aqs_index++)); do
  aqs_current_group="${aqs_groups[$aqs_index]}"
  aqs_current_output="${aqs_outputs[$aqs_index]}"
  aqs_check_gpu
  # Atomic directory creation protects against a concurrently created result.
  mkdir "$aqs_current_output"
  aqs_make_command
  printf '\n开始[%s/%s] %s：独立从头训练110轮，seed=0\n' \
    "$((aqs_index + 1))" "${#aqs_groups[@]}" "$aqs_current_group"
  "${aqs_cmd[@]}" 2>&1 | tee "$aqs_current_output/console.log"
  test -s "$aqs_current_output/best_stg2.pth" || aqs_fail "训练进程正常返回但未找到best_stg2.pth：$aqs_current_output"
  printf 'group=%s\ncompleted_utc=%s\n' "$aqs_current_group" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$aqs_current_output/aqs_sweep_completed.txt"
  printf '完成：%s；输出保留在：%s\n' "$aqs_current_group" "$aqs_current_output"
done
printf '\n所选%s组训练全部完成。\n' "${#aqs_groups[@]}"
