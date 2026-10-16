#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
serf_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$serf_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
serf_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; serf_dry=1
fi
serf_target="${SCBS_SERF_SSH_TARGET:-a5@100.76.151.67}"
serf_remote='/home/a5/MyProject/wq_project/D-FINE'
serf_manifest="$serf_local/tools/diagnostics/scbs_AQS_SERF_sync_files.txt"
while IFS= read -r serf_file; do
  [[ -z "$serf_file" ]] || test -f "$serf_local/$serf_file"
done < "$serf_manifest"
ssh "$serf_target" bash -s -- "$serf_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in train.py configs/dfine/dfine_s_scbs_hrw_110.yml \
    configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml \
    src/zoo/dfine/medium_finegrained.py src/zoo/dfine/semantic_differential_relation_fusion.py \
    src/zoo/dfine/scale_aligned_detail_fusion.py src/solver/query_diagnostics.py \
    src/solver/aqs_parameters.py tools/diagnostics/analyze_detection_diagnostics.py \
    tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dependency" || { printf '缺少原有依赖：%s\n' "$dependency" >&2; exit 1; }
done
# Never overwrite model sources underneath an existing D-FINE job.
python3 - "$1" <<'PY'
import os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
    try:
        texts = [arg.decode(errors='replace') for arg in (proc/'cmdline').read_bytes().split(b'\0') if arg]
        cwd = (proc/'cwd').resolve(strict=True)
    except (OSError, RuntimeError): continue
    scoped = cwd == root or root in cwd.parents or any(str(root) in arg for arg in texts)
    # python queue/eval/test/probe scripts inside this project, or train.py.
    active = any('train.py' in arg or ('tools/diagnostics/' in arg and arg.endswith('.py')) for arg in texts)
    if scoped and active:
        raise SystemExit(f'D-FINE任务仍在运行，暂不同步源码：PID={proc.name}')
PY
REMOTE
serf_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-SERF-backup-$(date +%Y%m%dT%H%M%S)-$$"
serf_cmd=(rsync -av --relative --files-from="$serf_manifest" --backup --backup-dir="$serf_backup")
[[ "$serf_dry" == 0 ]] || serf_cmd+=(--dry-run)
"${serf_cmd[@]}" "$serf_local/" "$serf_target:$serf_remote/"
printf 'SERF桌面源码同步完成；不修改原基线配置、数据、权重或旧输出。\n'
[[ "$serf_dry" == 1 ]] || printf '被替换源码备份：%s\n' "$serf_backup"
