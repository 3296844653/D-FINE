#!/usr/bin/env bash
# Whitelist sync only; never sync data/checkpoints/results, never kill jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
box_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$box_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
box_dry=0; box_wait=0
for box_arg in "$@"; do
  case "$box_arg" in
    --dry-run) box_dry=1 ;;
    --wait) box_wait=1 ;;
    *) printf '用法：bash tools/diagnostics/sync_scbs_boxlocal_five.sh [--dry-run] [--wait]\n' >&2; exit 1 ;;
  esac
done
box_manifest="$box_local/tools/diagnostics/scbs_boxlocal_five_sync_files.txt"
while IFS= read -r box_file; do
  [[ -n "$box_file" && -f "$box_local/$box_file" ]] || { printf '白名单文件缺失：%s\n' "$box_file" >&2; exit 1; }
done < "$box_manifest"
box_target="${SCBS_BOXL_SSH_TARGET:-a5@100.76.151.67}"
box_remote='/home/a5/MyProject/wq_project/D-FINE'
box_seed='configs/dfine/dfine_s_scbs_hrw_110_night_n4.yml'
box_backup="/home/a5/MyProject/wq_project/D-FINE-BoxLocal-five-backup-$(date +%Y%m%dT%H%M%S)-$$"
if [[ "$box_dry" == 1 ]]; then
  printf '仅本地演练：不连接服务器、不修改文件。白名单：\n'
  while IFS= read -r box_file; do printf '  %s\n' "$box_file"; done < "$box_manifest"
  printf '基础配置：%s（仅缺失时补齐；已有文件核对SHA256，不覆盖）。\n' "$box_seed"
  printf '目标：%s:%s/\n共享源码空闲等待=%s；不覆盖数据、权重和旧输出。\n' "$box_target" "$box_remote" "$box_wait"
  exit 0
fi
box_seed_sha="$(python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$box_local/$box_seed")"
ssh "$box_target" bash -s -- "$box_remote" "$box_wait" "$box_seed_sha" <<'REMOTE'
set -euo pipefail
cd "$1"
python3 - "$1" "$2" "$3" <<'PY'
import ast
import hashlib
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = ['train.py', 'configs/dfine/dfine_s_scbs_hrw_110.yml',
    'tools/diagnostics/wait_for_gpu.py', 'tools/diagnostics/analyze_detection_diagnostics.py',
    'src/solver/confusion_export.py', 'src/zoo/dfine/subpixel_context_fusion.py',
    'src/zoo/dfine/spatial_evidence_relation_fusion.py', 'src/zoo/dfine/night_probes.py',
    'src/zoo/dfine/finegrain_trials.py', 'src/zoo/dfine/discriminative_evidence.py',
    'src/zoo/dfine/behavior_refinement.py', 'src/zoo/dfine/instance_evidence.py',
    'src/zoo/dfine/rw_local_detail.py', 'src/zoo/dfine/rw_query_calibration.py',
    'src/zoo/dfine/evidence_contrast_aqs.py']
missing = [name for name in required if not (root/name).is_file()]
if missing:
    raise SystemExit('服务器缺少以下已有依赖：\n  ' + '\n  '.join(missing) + '\n尚未同步任何文件')

seed = root/'configs/dfine/dfine_s_scbs_hrw_110_night_n4.yml'
def check_seed():
    if seed.is_symlink():
        raise SystemExit(f'基础配置是符号链接，保留且不同步：{seed}')
    if seed.exists() and (not seed.is_file() or hashlib.sha256(seed.read_bytes()).hexdigest() != sys.argv[3]):
        raise SystemExit(f'服务器已有基础配置与桌面版本不同，保留原文件：{seed}；尚未同步任何文件，请先核对差异')

check_seed()
tree = ast.parse((root/'src/zoo/dfine/night_probes.py').read_text())
owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoxLocalQueryAdapter')
init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
args = {a.arg for a in [*init.args.args, *init.args.kwonlyargs]}
if not {'dim', 'grid_size', 'offset_scale', 'residual_scale', 'record_settings'} <= args:
    raise SystemExit('服务器BoxLocal参数接口过旧；先同步原BoxLocal参数扫描源码，本次未修改文件')
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
                         '；加 --wait 可等待结束后自动同步')
    print('等待D-FINE任务结束后同步；30秒复查，PID=' + ','.join(active), flush=True)
    time.sleep(30)
check_seed()  # GPU/source wait may be long; do not trust an earlier checksum.
print('共享源码使用检查通过；基础配置缺失时将自动补齐，已有文件不覆盖。', flush=True)
PY
REMOTE
# Bootstrap ONLY this missing dependency. Main rsync excludes it, so an
# existing baseline is never replaced, even if timestamps differ.
# Use a genuinely relative source: macOS openrsync does not honour the /./
# cut-point in an absolute --relative source (it would copy /Users/... too).
rsync -av --relative --ignore-existing "./$box_seed" "$box_target:$box_remote/"
rsync -av --checksum --relative --files-from="$box_manifest" --backup --backup-dir="$box_backup" \
  --exclude="/$box_seed" \
  "$box_local/" "$box_target:$box_remote/"
ssh "$box_target" bash -s -- "$box_remote" <<'VERIFY_REMOTE'
set -euo pipefail
python3 - "$1" <<'VERIFY_PY'
import socket
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root/'tools/diagnostics/scbs_boxlocal_five_sync_files.txt'
if not manifest.is_file():
    raise SystemExit(f'同步后缺少清单：{manifest}；不能开始训练，请保留同步完整输出')
files = manifest.read_text().splitlines()
launcher = 'tools/diagnostics/run_scbs_boxlocal_five_110.sh'
required = set(files) | {launcher, 'tools/diagnostics/scbs_boxlocal_five.py',
                         'tools/diagnostics/boxlocal_five_spec.py'}
missing = [name for name in sorted(required) if not (root/name).is_file() or not (root/name).stat().st_size]
if missing:
    raise SystemExit('同步后文件缺失或为空：\n  ' + '\n  '.join(missing) + '\n不能开始训练，请保留同步完整输出')
subprocess.run(['bash', '-n', str(root/launcher)], check=True)
print(f'已核对服务器：{socket.gethostname()}；项目目录：{root}', flush=True)
print(f'启动脚本已核对：{root/launcher}', flush=True)
VERIFY_PY
VERIFY_REMOTE
printf 'BoxLocal-L2 五组扩展同步完成。\n被替换文件备份：%s\n' "$box_backup"
