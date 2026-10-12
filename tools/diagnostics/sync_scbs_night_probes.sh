#!/usr/bin/env bash
# Explicit code/config allowlist only; no dataset/results/.git/delete operations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
night_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_night_probes.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数：%s\n' "$1" >&2; exit 1; }
  night_sync_dry=1
fi
night_sync_target="${NIGHT_SSH_TARGET:-a5@100.76.151.67}"
night_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
night_sync_manifest='tools/diagnostics/scbs_night_sync_files.txt'
while IFS= read -r night_sync_file; do
  [[ -z "$night_sync_file" ]] && continue
  [[ -f "$night_sync_file" ]] || { printf '缺少本地文件：%s\n' "$night_sync_file" >&2; exit 1; }
done < "$night_sync_manifest"
# Leave server's base config/data paths alone. It must already exist.
ssh "$night_sync_target" "test -f '$night_sync_remote/configs/dfine/dfine_s_scbs_hrw_110.yml'" || {
  printf '服务器未连接或缺少110轮基线配置；未同步。\n' >&2; exit 1;
}
night_sync_backup="/home/a5/MyProject/wq_project/D-FINE-night-probes-backup-$(date +%Y%m%dT%H%M%S)"
night_sync_cmd=(rsync -av --relative --files-from="$night_sync_manifest" --backup --backup-dir="$night_sync_backup")
[[ "$night_sync_dry" == 0 ]] || night_sync_cmd+=(--dry-run)
"${night_sync_cmd[@]}" ./ "$night_sync_target:$night_sync_remote/"
if [[ "$night_sync_dry" == 1 ]]; then
  printf '演练完成；未同步/覆盖服务器文件。\n'
else
  printf '白名单文件同步完成；被替换的服务器旧文件备份位置：%s\n' "$night_sync_backup"
  printf '未覆盖服务器基线配置、数据集或训练输出；未启动训练。\n'
fi
