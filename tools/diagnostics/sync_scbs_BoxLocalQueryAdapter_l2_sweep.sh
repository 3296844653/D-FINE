#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
box_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$box_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
box_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; box_dry=1
fi
box_target="${SCBS_BOX_SSH_TARGET:-a5@100.76.151.67}"
box_remote='/home/a5/MyProject/wq_project/D-FINE'
box_manifest="$box_local/tools/diagnostics/scbs_BoxLocalQueryAdapter_l2_sweep_sync_files.txt"
while IFS= read -r box_file; do
  [[ -z "$box_file" ]] || test -f "$box_local/$box_file"
done < "$box_manifest"
ssh "$box_target" bash -s -- "$box_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in train.py configs/dfine/dfine_s_scbs_hrw_110.yml \
    src/zoo/dfine/medium_finegrained.py src/zoo/dfine/semantic_differential_relation_fusion.py \
    src/zoo/dfine/scale_aligned_detail_fusion.py src/solver/query_diagnostics.py \
    tools/diagnostics/analyze_detection_diagnostics.py; do
  test -f "$dependency" || { printf '缺少现有依赖：%s\n' "$dependency" >&2; exit 1; }
done
python3 - "$1" <<'PY'
import os,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        texts=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        cwd=(proc/'cwd').resolve(strict=True)
    except (OSError,RuntimeError):continue
    scoped=cwd==root or root in cwd.parents or any(str(root) in s for s in texts)
    active=any('train.py' in s or ('tools/diagnostics/' in s and s.endswith('.py')) for s in texts)
    if scoped and active:raise SystemExit(f'D-FINE任务仍在运行，暂不同步共享源码：PID={proc.name}')
PY
REMOTE
box_backup="/home/a5/MyProject/wq_project/D-FINE-BoxLocalSweep-backup-$(date +%Y%m%dT%H%M%S)-$$"
box_cmd=(rsync -av --relative --files-from="$box_manifest" --backup --backup-dir="$box_backup")
[[ "$box_dry" == 0 ]] || box_cmd+=(--dry-run)
"${box_cmd[@]}" "$box_local/" "$box_target:$box_remote/"
printf '桌面BoxLocal参数接口和新配置同步完成；原基线配置、数据及旧输出不修改。\n'
[[ "$box_dry" == 1 ]] || printf '被替换源码备份：%s\n' "$box_backup"
