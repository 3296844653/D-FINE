#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
scf_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$scf_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
scf_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; scf_dry=1
fi
scf_target="${SCBS_SCF_SSH_TARGET:-a5@100.76.151.67}"
scf_remote='/home/a5/MyProject/wq_project/D-FINE'
scf_manifest="$scf_local/tools/diagnostics/scbs_AQS_SubpixelContextFusion_sync_files.txt"
while IFS= read -r scf_file; do
  [[ -z "$scf_file" ]] || test -f "$scf_local/$scf_file"
done < "$scf_manifest"
ssh "$scf_target" bash -s -- "$scf_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dep in train.py configs/dfine/dfine_s_scbs_hrw_110.yml \
    src/zoo/dfine/semantic_differential_relation_fusion.py src/zoo/dfine/scale_aligned_detail_fusion.py \
    src/solver/query_diagnostics.py src/solver/aqs_parameters.py \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dep" || { printf '缺少现有依赖：%s\n' "$dep" >&2; exit 1; }
done
# Do not replace model source during any D-FINE training/evaluation.
if ps -ww -u "$(id -un)" -o args= | grep -E '[p]ython[^ ]* .*train\.py|[t]orchrun .*train\.py|[t]orchrun .*eval_subpixel_context\.py' >/dev/null; then
  printf 'D-FINE仍有训练/评估进程；为避免源码与权重不一致，暂不同步。\n' >&2
  exit 1
fi
REMOTE
scf_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-SubpixelContextFusion-backup-$(date +%Y%m%dT%H%M%S)-$$"
scf_cmd=(rsync -av --relative --files-from="$scf_manifest" --backup --backup-dir="$scf_backup")
[[ "$scf_dry" == 0 ]] || scf_cmd+=(--dry-run)
"${scf_cmd[@]}" "$scf_local/" "$scf_target:$scf_remote/"
printf '桌面源码同步完成%s；不修改数据、基线配置、权重或旧输出。\n' "$( [[ "$scf_dry" == 1 ]] && printf '（演练）' || printf '')"
[[ "$scf_dry" == 1 ]] || printf '被替换源码备份：%s\n' "$scf_backup"
