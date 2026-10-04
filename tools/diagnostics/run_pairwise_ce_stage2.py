"""Prepare and launch a guarded shared-prefix stage-2 weight comparison.

No solver, metric, dataset, or checkpoint-state mutation. Both profiles resume
the SAME full CE=0.1 first-stage checkpoint, including model AND EMA/optimizer.
Only use trusted checkpoints produced by your own training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.core.yaml_utils import load_config

CONFIGS = {
    "w01": "configs/dfine/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce_stage2_w01.yml",
    "w005": "configs/dfine/dfine_s_scb3s_3cls_bs32_rw_pairwise_ce_stage2_w005.yml",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(state):
    required = {
        "last_epoch", "model", "ema", "optimizer", "criterion",
        "lr_scheduler", "lr_warmup_scheduler", "scaler",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        missing = sorted(required - set(state)) if isinstance(state, dict) else sorted(required)
        raise ValueError(f"Not a full training checkpoint; missing states: {missing}")
    if type(state["last_epoch"]) is not int or state["last_epoch"] != 119:
        raise ValueError(
            f"Expected CE=0.1 best_stg1 last_epoch=119, got {state['last_epoch']!r}. "
            "Do not change its epoch by hand or use best_stg2/short20 weights."
        )
    for key in required - {"last_epoch"}:
        if not isinstance(state[key], dict):
            raise ValueError(f"Invalid {key} state")
    if not isinstance(state["ema"].get("module"), dict):
        raise ValueError("Missing EMA module weights")
    if state["ema"].get("updates", 0) <= 0:
        raise ValueError("EMA must contain its existing update count")
    for weights in (state["model"], state["ema"]["module"]):
        heads = [v for k, v in weights.items() if k.startswith("decoder.dec_score_head.")
                 and k.endswith(".weight")]
        if len(heads) != 3 or any(v.shape != (3, 256) for v in heads):
            raise ValueError("Expected a 3-class D-FINE-S checkpoint with 3 decoder score heads")
    optimizer = state["optimizer"]
    if not optimizer.get("param_groups") or not optimizer.get("state"):
        raise ValueError("Missing populated optimizer state; this would not preserve training history")
    warmup = state["lr_warmup_scheduler"]
    if warmup.get("last_step", -1) < warmup.get("warmup_duration", 500):
        raise ValueError("Source warmup is not finished")
    return {
        "last_epoch": state["last_epoch"],
        "ema_updates": state["ema"]["updates"],
        "optimizer_lrs": [group["lr"] for group in optimizer["param_groups"]],
        "scheduler_last_epoch": state["lr_scheduler"].get("last_epoch"),
        "warmup_last_step": warmup["last_step"],
    }


def prepare_run(checkpoint, output_dir, profile, seed=0, master_port=7777, use_amp=False):
    source = Path(checkpoint).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    cfg_path = ROOT / CONFIGS[profile]
    cfg = load_config(str(cfg_path), cfg={})
    expected_weight = 0.1 if profile == "w01" else 0.05
    assert cfg["epochs"] == 132
    assert cfg["num_classes"] == 3
    assert cfg["DFINECriterion"]["pairwise_ce_weight"] == expected_weight
    assert cfg["train_dataloader"]["collate_fn"]["stop_epoch"] == 120
    assert cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == 120
    # Never merge into an existing experiment or overwrite its checkpoint/logs.
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a NEW run name.")
    source_hash = sha256_file(source)
    state = torch.load(source, map_location="cpu", weights_only=False)
    summary = validate_checkpoint(state)
    del state
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "best_stg1.pth"
    shutil.copy2(source, prepared)
    if sha256_file(prepared) != source_hash:
        raise RuntimeError("Copied checkpoint hash differs; do not start training")
    command = [
        "torchrun", f"--master_port={master_port}", "--nproc_per_node=1",
        str(ROOT / "train.py"), "-c", str(cfg_path), "-r", str(prepared),
        "--output-dir", str(output), f"--seed={seed}",
    ]
    if use_amp:
        command.append("--use-amp")
    metadata = {
        "profile": profile, "pairwise_ce_weight": expected_weight,
        "source_checkpoint": str(source), "source_sha256": source_hash,
        "prepared_checkpoint": str(prepared), "start_epoch": 120,
        "end_epoch_inclusive": 131, "training_epochs": 12,
        "source_states": summary, "seed": seed, "use_amp": use_amp,
        "config": str(cfg_path), "command": command,
        "scope": "CE=0.1 shared first-stage prefix; compare only last-stage CE weights",
        "note": "Original solver can reload best_stg1 during EMA refresh. RNG state is not "
                "restored by the original checkpoint loader; this is a paired fork, not "
                "bitwise reproduction of the old complete run.",
    }
    (output / "stage2_source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Original CE=0.1 best_stg1.pth")
    parser.add_argument("--output-dir", required=True, help="A NEW output directory")
    parser.add_argument("--profile", choices=CONFIGS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=7777)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/copy without training")
    args = parser.parse_args()
    metadata = prepare_run(
        args.checkpoint, args.output_dir, args.profile,
        args.seed, args.master_port, args.use_amp,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print("Prepared full checkpoint. Resume epochs 120..131, no fresh optimizer or EMA.", flush=True)
    if not args.prepare_only:
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        result = subprocess.run(metadata["command"], cwd=ROOT, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
        print("Stage-2 run finished. Preserve log.txt, stage2_source.json and checkpoints.", flush=True)
        if not (Path(args.output_dir) / "best_stg2.pth").is_file():
            print("No best_stg2.pth: check log.txt; original solver only saves it when its "
                  "best-stat rules allow. This is not evidence that training did not run.", flush=True)


if __name__ == "__main__":
    main()
