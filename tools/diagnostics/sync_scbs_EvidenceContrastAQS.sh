#!/usr/bin/env bash
# Run from authoritative Desktop checkout; preserve server baseline/data/results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
eca_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_EvidenceContrastAQS.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数：%s\n' "$1" >&2; exit 1; }
  eca_sync_dry=1
fi
eca_sync_target="${SCBS_ECAQS_SSH_TARGET:-a5@100.76.151.67}"
eca_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
eca_sync_manifest='tools/diagnostics/scbs_EvidenceContrastAQS_sync_files.txt'
while IFS= read -r eca_sync_file; do
  [[ -z "$eca_sync_file" ]] && continue
  [[ -f "$eca_sync_file" ]] || { printf '缺少本地文件：%s\n' "$eca_sync_file" >&2; exit 1; }
done < "$eca_sync_manifest"
# Staging contains an unfinished unrelated literature branch. Do not distribute it.
if rg -q literature_modules src/zoo/dfine/dfine.py; then
  printf '请从/Users/wq/Desktop/D-FINE执行；暂存工作区含未完成的其他实验，未同步。\n' >&2
  exit 1
fi
ssh "$eca_sync_target" "test -f '$eca_sync_remote/configs/dfine/dfine_s_scbs_hrw_110.yml' && test -f '$eca_sync_remote/src/zoo/dfine/night_probes.py' && test -f '$eca_sync_remote/src/zoo/dfine/medium_finegrained.py' && test -f '$eca_sync_remote/src/solver/aqs_parameters.py'" || {
  printf '服务器未连接或缺少现有基线/模块依赖；未同步。\n' >&2; exit 1;
}
eca_sync_backup="/home/a5/MyProject/wq_project/D-FINE-EvidenceContrastAQS-backup-$(date +%Y%m%dT%H%M%S)"
eca_sync_command=(rsync -av --relative --files-from="$eca_sync_manifest" --backup --backup-dir="$eca_sync_backup")
[[ "$eca_sync_dry" == 0 ]] || eca_sync_command+=(--dry-run)
"${eca_sync_command[@]}" ./ "$eca_sync_target:$eca_sync_remote/"
if [[ "$eca_sync_dry" == 1 ]]; then
  printf 'EvidenceContrastAQS同步演练完成；未替换服务器文件。\n'
else
  printf 'EvidenceContrastAQS同步完成；被替换文件可从以下备份恢复：%s\n' "$eca_sync_backup"
  printf '没有覆盖基线配置、数据集、旧输出；没有启动训练。\n'
fi
