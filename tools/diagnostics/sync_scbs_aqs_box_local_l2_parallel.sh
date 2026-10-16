#!/usr/bin/env bash
# This version changes two shared model files. Wait is optional; never kill jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
parallel_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$parallel_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
parallel_dry=0; parallel_wait=0
for parallel_arg in "$@"; do
  case "$parallel_arg" in
    --dry-run) parallel_dry=1 ;;
    --wait) parallel_wait=1 ;;
    *) printf '用法：bash tools/diagnostics/sync_scbs_aqs_box_local_l2_parallel.sh [--dry-run] [--wait]\n' >&2; exit 1 ;;
  esac
done
parallel_manifest="$parallel_local/tools/diagnostics/scbs_aqs_box_local_l2_parallel_sync_files.txt"
while IFS= read -r parallel_file; do
  [[ -n "$parallel_file" && -f "$parallel_local/$parallel_file" ]] || { printf '白名单文件缺失：%s\n' "$parallel_file" >&2; exit 1; }
done < "$parallel_manifest"
parallel_target="${SCBS_COMBO_SSH_TARGET:-a5@100.76.151.67}"
parallel_remote='/home/a5/MyProject/wq_project/D-FINE'
parallel_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-BoxLocal-parallel-backup-$(date +%Y%m%dT%H%M%S)-$$"
if [[ "$parallel_dry" == 1 ]]; then
  printf '仅本地演练：不连接服务器、不修改文件。白名单：\n'
  while IFS= read -r parallel_file; do printf '  %s\n' "$parallel_file"; done < "$parallel_manifest"
  printf '目标：%s:%s/\n共享源码空闲等待=%s；不覆盖数据、权重和旧输出。\n' "$parallel_target" "$parallel_remote" "$parallel_wait"
  exit 0
fi
ssh "$parallel_target" bash -s -- "$parallel_remote" "$parallel_wait" <<'REMOTE'
set -euo pipefail
cd "$1"
python3 - "$1" "$2" <<'PY'
import ast
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = ['train.py', 'configs/dfine/dfine_s_scbs_hrw_110.yml',
            'tools/diagnostics/wait_for_gpu.py', 'tools/diagnostics/analyze_detection_diagnostics.py',
            'src/zoo/dfine/spatial_evidence_relation_fusion.py', 'src/zoo/dfine/night_probes.py',
            'src/zoo/dfine/finegrain_trials.py', 'src/zoo/dfine/discriminative_evidence.py',
            'src/zoo/dfine/behavior_refinement.py', 'src/zoo/dfine/instance_evidence.py',
            'src/zoo/dfine/rw_local_detail.py', 'src/zoo/dfine/rw_query_calibration.py',
            'src/zoo/dfine/evidence_contrast_aqs.py']
for name in required:
    if not (root/name).is_file():
        raise SystemExit(f'服务器缺少已有项目依赖：{name}；尚未同步任何文件')
tree = ast.parse((root/'src/zoo/dfine/night_probes.py').read_text())
owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoxLocalQueryAdapter')
init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
args = {a.arg for a in [*init.args.args, *init.args.kwonlyargs]}
if not {'dim', 'grid_size', 'offset_scale', 'residual_scale', 'record_settings'} <= args:
    raise SystemExit('服务器BoxLocal参数接口过旧；本同步未修改文件，请先完成原参数扫描源码同步')
while True:
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            words = [s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            cwd = (proc/'cwd').resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        scoped = cwd == root or root in cwd.parents or any(str(root) in s for s in words)
        uses_repo = any(Path(s).name == 'train.py' or
                        ('tools/diagnostics/' in s and s.endswith(('.py', '.sh'))) for s in words)
        if scoped and uses_repo:
            active.append(proc.name)
    if not active:
        break
    if sys.argv[2] != '1':
        raise SystemExit('D-FINE仍有任务，尚未同步共享源码：PID=' + ','.join(active) +
                         '；加 --wait 可自动等待结束后同步')
    print('等待D-FINE任务结束后同步；30秒复查，PID=' + ','.join(active), flush=True)
    time.sleep(30)
print('共享源码使用检查通过。', flush=True)
PY
REMOTE
rsync -av --relative --files-from="$parallel_manifest" --backup --backup-dir="$parallel_backup" \
  "$parallel_local/" "$parallel_target:$parallel_remote/"
printf 'AQS + BoxLocal-L2 并行版同步完成。\n被替换文件备份：%s\n' "$parallel_backup"
