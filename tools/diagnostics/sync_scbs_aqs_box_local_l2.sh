#!/usr/bin/env bash
# Configuration/tools only: no shared model source or baseline YAML is replaced.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
combo_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$combo_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 && "${1:---dry-run}" == --dry-run ]] || {
  printf '用法：bash tools/diagnostics/sync_scbs_aqs_box_local_l2.sh [--dry-run]\n' >&2; exit 1;
}
combo_manifest="$combo_local/tools/diagnostics/scbs_aqs_box_local_l2_sync_files.txt"
while IFS= read -r combo_file; do
  [[ -z "$combo_file" ]] || test -f "$combo_local/$combo_file"
done < "$combo_manifest"
combo_target="${SCBS_COMBO_SSH_TARGET:-a5@100.76.151.67}"
combo_remote='/home/a5/MyProject/wq_project/D-FINE'
combo_backup="/home/a5/MyProject/wq_project/D-FINE-BoxLocal-AQS-config-backup-$(date +%Y%m%dT%H%M%S)-$$"
combo_command=(rsync -av --relative --files-from="$combo_manifest" --backup --backup-dir="$combo_backup")
if [[ "${1:-}" == --dry-run ]]; then
  printf '仅本地演练：不连接服务器、不修改任何文件。白名单：\n'
  while IFS= read -r combo_file; do printf '  %s\n' "$combo_file"; done < "$combo_manifest"
  printf '目标：%s:%s/\n不会同步模型源码、数据、权重、原基线配置或旧输出。\n' "$combo_target" "$combo_remote"
  exit 0
fi
ssh "$combo_target" bash -s -- "$combo_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in train.py configs/dfine/dfine_s_scbs_hrw_110.yml \
    tools/diagnostics/wait_for_gpu.py tools/diagnostics/analyze_detection_diagnostics.py; do
  test -f "$dependency" || { printf '缺少已有项目依赖：%s\n' "$dependency" >&2; exit 1; }
done
python3 - "$1" <<'PY'
import ast, os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
def args(file, cls, method):
    tree = ast.parse((root/file).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    func = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return {a.arg for a in [*func.args.args, *func.args.kwonlyargs]}
required = {'allow_aqs_l2_local_combo', 'box_local_dim', 'box_local_grid_size',
            'box_local_offset_scale', 'box_local_residual_scale'}
assert required <= args('src/zoo/dfine/dfine.py', 'DFINE', '__init__'), '服务器缺少既有BoxLocal组合接口；本同步不会覆盖模型源码'
assert {'allow_aqs_l2_local_combo', 'box_local_options'} <= args(
    'src/zoo/dfine/dfine_decoder.py', 'DFINETransformer', 'configure_night_probe'), '服务器Decoder组合接口不兼容'
config = 'dfine_s_scbs_hrw_110_AQS_BoxLocalQueryAdapter_l2.yml'
launcher = 'run_scbs_aqs_box_local_l2_110.sh'
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        words = [s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        cwd = (proc/'cwd').resolve(strict=True)
    except (OSError, RuntimeError):
        continue
    scoped = cwd == root or root in cwd.parents
    if scoped and any(Path(word).name in (config, launcher) for word in words):
        raise SystemExit(f'此组合任务仍在运行，保留其配置/脚本：PID={proc.name}')
print('既有模型接口检查通过；其他实验可继续运行，不替换共享源码。')
PY
REMOTE
"${combo_command[@]}" "$combo_local/" "$combo_target:$combo_remote/"
printf 'BoxLocal-L2 + AQS 配置/启动工具同步完成。\n被替换文件备份：%s\n' "$combo_backup"
