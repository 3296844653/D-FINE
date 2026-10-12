#!/usr/bin/env bash
# Explicit Desktop source; never sync a Codex worktree, data or training results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
evidence_sync_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$evidence_sync_local" ]] || {
  printf '只允许从 /Users/wq/Desktop/D-FINE 同步。\n' >&2; exit 1;
}
evidence_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_AQS_instance_evidence.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }
  evidence_sync_dry=1
fi
evidence_sync_target="${SCBS_EVIDENCE_SSH_TARGET:-a5@100.76.151.67}"
evidence_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
evidence_sync_manifest="$evidence_sync_local/tools/diagnostics/scbs_AQS_instance_evidence_sync_files.txt"
while IFS= read -r evidence_sync_file; do
  [[ -z "$evidence_sync_file" ]] && continue
  [[ -f "$evidence_sync_local/$evidence_sync_file" ]] || { printf '缺少本地文件：%s\n' "$evidence_sync_file" >&2; exit 1; }
done < "$evidence_sync_manifest"
printf '本地来源：%s\n服务器目标：%s:%s\n' "$evidence_sync_local" "$evidence_sync_target" "$evidence_sync_remote"
ssh "$evidence_sync_target" bash -s -- "$evidence_sync_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in train.py configs/dfine/dfine_s_scbs_hrw_110.yml \
    src/zoo/dfine/hybrid_encoder.py src/zoo/dfine/medium_finegrained.py \
    src/zoo/dfine/scale_aligned_detail_fusion.py src/zoo/dfine/semantic_differential_relation_fusion.py \
    src/solver/query_diagnostics.py src/solver/aqs_parameters.py \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dependency" || { printf '缺少现有服务器依赖：%s；停止同步。\n' "$dependency" >&2; exit 1; }
done
REMOTE
evidence_sync_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-instance-evidence-backup-$(date +%Y%m%dT%H%M%S)-$$"
evidence_sync_command=(rsync -av --relative --files-from="$evidence_sync_manifest" --backup --backup-dir="$evidence_sync_backup")
[[ "$evidence_sync_dry" == 0 ]] || evidence_sync_command+=(--dry-run)
"${evidence_sync_command[@]}" "$evidence_sync_local/" "$evidence_sync_target:$evidence_sync_remote/"
if [[ "$evidence_sync_dry" == 1 ]]; then
  printf '同步演练完成，没有替换文件。\n'
else
  printf '同步完成；替换源码的备份：%s\n未覆盖基线配置、数据、权重或旧结果，未启动训练。\n' "$evidence_sync_backup"
fi
