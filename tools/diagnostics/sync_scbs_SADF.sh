#!/usr/bin/env bash
# Run from authoritative Desktop checkout; preserve server baseline/data/results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sadf_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_SADF.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数：%s\n' "$1" >&2; exit 1; }
  sadf_sync_dry=1
fi
sadf_sync_target="${SCBS_SADF_SSH_TARGET:-a5@100.76.151.67}"
sadf_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
sadf_sync_manifest='tools/diagnostics/scbs_SADF_sync_files.txt'
while IFS= read -r sadf_sync_file; do
  [[ -z "$sadf_sync_file" ]] && continue
  [[ -f "$sadf_sync_file" ]] || { printf '缺少本地文件：%s\n' "$sadf_sync_file" >&2; exit 1; }
done < "$sadf_sync_manifest"
# Staging contains an unfinished unrelated literature branch. Do not distribute it.
if rg -q literature_modules src/zoo/dfine/dfine.py; then
  printf '请从/Users/wq/Desktop/D-FINE执行；暂存工作区含未完成的其他实验，未同步。\n' >&2
  exit 1
fi
ssh "$sadf_sync_target" "test -f '$sadf_sync_remote/configs/dfine/dfine_s_scbs_hrw_110.yml' && test -f '$sadf_sync_remote/src/zoo/dfine/night_probes.py' && test -f '$sadf_sync_remote/src/zoo/dfine/medium_finegrained.py'" || {
  printf '服务器未连接或缺少现有基线/模块依赖；未同步。\n' >&2; exit 1;
}
sadf_sync_backup="/home/a5/MyProject/wq_project/D-FINE-SADF-backup-$(date +%Y%m%dT%H%M%S)"
sadf_sync_command=(rsync -av --relative --files-from="$sadf_sync_manifest" --backup --backup-dir="$sadf_sync_backup")
[[ "$sadf_sync_dry" == 0 ]] || sadf_sync_command+=(--dry-run)
"${sadf_sync_command[@]}" ./ "$sadf_sync_target:$sadf_sync_remote/"
if [[ "$sadf_sync_dry" == 1 ]]; then
  printf 'SADF同步演练完成；未替换服务器文件。\n'
else
  printf 'SADF同步完成；被替换文件可从以下备份恢复：%s\n' "$sadf_sync_backup"
  printf '没有覆盖基线配置、数据集、旧输出；没有启动训练。\n'
fi
