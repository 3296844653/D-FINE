#!/usr/bin/env bash
# Explicit Desktop source, scoped rsync, recoverable remote source backups.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sdrf_sync_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$sdrf_sync_local" ]] || {
  printf '请使用 /Users/wq/Desktop/D-FINE，不同步Codex暂存工作区。\n' >&2; exit 1;
}
sdrf_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_SDRF.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数：%s\n' "$1" >&2; exit 1; }
  sdrf_sync_dry=1
fi
sdrf_sync_target="${SCBS_SDRF_SSH_TARGET:-a5@100.76.151.67}"
sdrf_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
sdrf_sync_manifest="$sdrf_sync_local/tools/diagnostics/scbs_SDRF_sync_files.txt"
while IFS= read -r sdrf_sync_file; do
  [[ -z "$sdrf_sync_file" ]] && continue
  [[ -f "$sdrf_sync_local/$sdrf_sync_file" ]] || { printf '缺少本地文件：%s\n' "$sdrf_sync_file" >&2; exit 1; }
done < "$sdrf_sync_manifest"
printf '本地来源：%s\n服务器目标：%s:%s\n' "$sdrf_sync_local" "$sdrf_sync_target" "$sdrf_sync_remote"
ssh "$sdrf_sync_target" bash -s -- "$sdrf_sync_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in configs/dfine/dfine_s_scbs_hrw_110.yml src/zoo/dfine/night_probes.py \
    src/zoo/dfine/finegrain_trials.py src/zoo/dfine/medium_finegrained.py \
    src/zoo/dfine/scale_aligned_detail_fusion.py tools/diagnostics/analyze_detection_diagnostics.py; do
  test -f "$dependency" || { printf '缺少现有依赖：%s；停止同步。\n' "$dependency" >&2; exit 1; }
done
REMOTE
sdrf_sync_backup="/home/a5/MyProject/wq_project/D-FINE-SDRF-backup-$(date +%Y%m%dT%H%M%S)-$$"
sdrf_sync_command=(rsync -av --relative --files-from="$sdrf_sync_manifest" --backup --backup-dir="$sdrf_sync_backup")
[[ "$sdrf_sync_dry" == 0 ]] || sdrf_sync_command+=(--dry-run)
"${sdrf_sync_command[@]}" "$sdrf_sync_local/" "$sdrf_sync_target:$sdrf_sync_remote/"
if [[ "$sdrf_sync_dry" == 1 ]]; then
  printf 'SDRF同步演练完成，未替换服务器文件。\n'
else
  printf 'SDRF同步完成；被替换源码的备份：%s\n没有覆盖基线配置、数据集和旧结果，没有启动训练。\n' "$sdrf_sync_backup"
fi
