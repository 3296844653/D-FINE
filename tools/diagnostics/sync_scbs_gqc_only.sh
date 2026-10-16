#!/usr/bin/env bash
# Whitelisted code/config only. No baseline overwrite, data, weights or old output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
gqc_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$gqc_local" ]] || { printf '只允许从Mac桌面D-FINE同步。\n' >&2; exit 1; }
gqc_dry=0; gqc_wait=0
for gqc_arg in "$@"; do
  case "$gqc_arg" in
    --dry-run) gqc_dry=1 ;;
    --wait) gqc_wait=1 ;;
    *) printf '用法：bash tools/diagnostics/sync_scbs_gqc_only.sh [--dry-run] [--wait]\n' >&2; exit 1 ;;
  esac
done
gqc_manifest="$gqc_local/tools/diagnostics/scbs_gqc_only_sync_files.txt"
while IFS= read -r gqc_file; do
  [[ -n "$gqc_file" && -f "$gqc_local/$gqc_file" ]] || { printf '白名单文件缺失：%s\n' "$gqc_file" >&2; exit 1; }
done < "$gqc_manifest"
gqc_target="${SCBS_GQC_SSH_TARGET:-a5@100.76.151.67}"
gqc_remote='/home/a5/MyProject/wq_project/D-FINE'
gqc_backup="/home/a5/MyProject/wq_project/D-FINE-GQC-only-backup-$(date +%Y%m%dT%H%M%S)-$$"
if [[ "$gqc_dry" == 1 ]]; then
  printf '仅本地演练：不连接服务器、不修改文件。白名单：\n'
  while IFS= read -r gqc_file; do printf '  %s\n' "$gqc_file"; done < "$gqc_manifest"
  printf '目标：%s:%s/\n不依赖night_n4或AQS配置；原始110基线须存在且不覆盖。共享源码等待=%s。\n' "$gqc_target" "$gqc_remote" "$gqc_wait"
  exit 0
fi
ssh "$gqc_target" bash -s -- "$gqc_remote" "$gqc_wait" <<'REMOTE'
set -euo pipefail
cd "$1"
python3 - "$1" "$2" <<'PY'
import os
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
required = ['train.py', 'configs/dfine/dfine_s_scbs_hrw_110.yml',
    'configs/dfine/dfine_hgnetv2_s_coco.yml', 'configs/dataset/coco_detection.yml', 'configs/runtime.yml',
    'configs/dfine/include/dataloader.yml', 'configs/dfine/include/optimizer.yml', 'configs/dfine/include/dfine_hgnetv2.yml',
    'src/solver/confusion_export.py', 'src/solver/det_solver.py', 'src/solver/det_engine.py',
    'src/zoo/dfine/hybrid_encoder.py', 'src/zoo/dfine/night_probes.py', 'src/zoo/dfine/finegrain_trials.py',
    'src/zoo/dfine/spatial_evidence_relation_fusion.py', 'src/zoo/dfine/subpixel_context_fusion.py',
    'src/zoo/dfine/discriminative_evidence.py', 'src/zoo/dfine/instance_evidence.py',
    'src/zoo/dfine/rw_local_detail.py', 'src/zoo/dfine/rw_query_calibration.py', 'src/zoo/dfine/evidence_contrast_aqs.py',
    'tools/diagnostics/analyze_detection_diagnostics.py']
missing = [name for name in required if not (root/name).is_file()]
if missing:
    raise SystemExit('服务器缺少以下已有项目依赖：\n  '+'\n  '.join(missing)+'\n尚未同步任何文件')
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
        raise SystemExit('D-FINE仍有任务，尚未同步共享源码：PID='+','.join(active)+'；可加 --wait 等待，不会结束任何任务')
    print('等待D-FINE任务结束后同步；30秒复查，PID='+','.join(active), flush=True)
    time.sleep(30)
print('源码使用检查通过；GQC单模块不依赖AQS或BoxLocal配置。', flush=True)
PY
REMOTE
rsync -av --checksum --relative --files-from="$gqc_manifest" --backup --backup-dir="$gqc_backup" \
  "$gqc_local/" "$gqc_target:$gqc_remote/"
ssh "$gqc_target" bash -s -- "$gqc_remote" <<'VERIFY_REMOTE'
set -euo pipefail
python3 - "$1" <<'VERIFY_PY'
import ast
import socket
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root/'tools/diagnostics/scbs_gqc_only_sync_files.txt'
if not manifest.is_file():
    raise SystemExit('同步后缺少清单；请保留完整同步输出，不要启动训练')
names = set(manifest.read_text().splitlines()) | {'tools/diagnostics/run_scbs_gqc_only_110.sh',
    'tools/diagnostics/scbs_gqc_only.py', 'configs/dfine/dfine_s_scbs_hrw_110_GeometryAwareQueryCompetition_only.yml'}
missing = [name for name in sorted(names) if not (root/name).is_file() or not (root/name).stat().st_size]
if missing:
    raise SystemExit('同步后文件缺失或为空：\n  '+'\n  '.join(missing)+'\n不能开始训练')
tree = ast.parse((root/'src/zoo/dfine/dfine_decoder.py').read_text())
owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DFINETransformer')
init = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
if 'allow_gqc_standalone' not in {arg.arg for arg in [*init.args.args, *init.args.kwonlyargs]}:
    raise SystemExit('解码器仍是旧版，GQC单模块接口未同步；不能开始训练')
launcher = root/'tools/diagnostics/run_scbs_gqc_only_110.sh'
subprocess.run(['bash', '-n', str(launcher)], check=True)
ast.parse((root/'tools/diagnostics/scbs_gqc_only.py').read_text())
print(f'已核对服务器：{socket.gethostname()}；项目目录：{root}', flush=True)
print(f'启动脚本已核对：{launcher}', flush=True)
VERIFY_PY
VERIFY_REMOTE
printf 'GQC单模块同步完成。\n被替换源码备份：%s\n' "$gqc_backup"
