#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
behavior_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$behavior_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
behavior_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; behavior_dry=1
fi
behavior_target="${SCBS_BEHAVIOR_SSH_TARGET:-a5@100.76.151.67}"
behavior_remote='/home/a5/MyProject/wq_project/D-FINE'
behavior_manifest="$behavior_local/tools/diagnostics/scbs_AQS_BehaviorFour_sync_files.txt"
while IFS= read -r behavior_file; do
  [[ -z "$behavior_file" ]] || test -f "$behavior_local/$behavior_file"
done < "$behavior_manifest"
ssh "$behavior_target" bash -s -- "$behavior_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in train.py configs/dfine/dfine_s_scbs_hrw_110.yml \
    configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml \
    src/zoo/dfine/medium_finegrained.py src/zoo/dfine/semantic_differential_relation_fusion.py \
    src/zoo/dfine/scale_aligned_detail_fusion.py src/solver/query_diagnostics.py \
    src/solver/aqs_parameters.py tools/diagnostics/analyze_detection_diagnostics.py \
    tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dependency" || { printf '缺少现有依赖：%s\n' "$dependency" >&2; exit 1; }
done
# Scope the guard to this project. Other projects (including llama) are untouched.
python3 - "$1" <<'PY'
import os,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        argv=(proc/'cmdline').read_bytes().split(b'\0')
        cwd=(proc/'cwd').resolve(strict=True)
    except (OSError,RuntimeError):continue
    texts=[arg.decode(errors='replace') for arg in argv if arg]
    scoped=cwd==root or root in cwd.parents or any(str(root) in arg for arg in texts)
    active=any('train.py' in arg or 'eval_behavior_refinement.py' in arg
               or 'scbs_behavior_four_queue.py' in arg or 'test_behavior_refinement.py' in arg for arg in texts)
    if scoped and active:
        raise SystemExit(f'D-FINE训练/评估/队列仍在运行，暂不同步源码：PID={proc.name}')
PY
REMOTE
behavior_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-BehaviorFour-backup-$(date +%Y%m%dT%H%M%S)-$$"
behavior_cmd=(rsync -av --relative --files-from="$behavior_manifest" --backup --backup-dir="$behavior_backup")
[[ "$behavior_dry" == 0 ]] || behavior_cmd+=(--dry-run)
"${behavior_cmd[@]}" "$behavior_local/" "$behavior_target:$behavior_remote/"
printf '桌面源码同步完成；不修改数据、原基线配置、权重或旧输出。\n'
[[ "$behavior_dry" == 1 ]] || printf '被替换源码备份：%s\n' "$behavior_backup"
