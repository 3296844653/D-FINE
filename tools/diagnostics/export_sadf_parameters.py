"""Read SADF settings and learned channel scales from a trusted own checkpoint.

No dataset/GPU/model import is needed. Fixed settings are stored in the SADF
state, so they are not guessed from a filename or supplied temperature.
"""
import argparse
import json
from pathlib import Path

import torch


def checkpoint_report(checkpoint, path):
    candidates = {"model": checkpoint.get("model"), "ema": checkpoint.get("ema")}
    sources = {}
    for name, state in candidates.items():
        if not isinstance(state, dict):
            continue
        if "module" in state:
            state = state["module"]
        for key, raw in state.items():
            if not key.endswith("encoder.sadf.raw_alpha"):
                continue
            prefix = key.removesuffix("raw_alpha")
            settings = state[prefix + "settings"].detach().float().cpu()
            if settings.numel() != 9 or int(settings[0]) != 1:
                raise ValueError("Unsupported SADF settings schema")
            alpha = settings[4] * raw.detach().float().cpu().tanh()
            if not torch.isfinite(alpha).all():
                raise ValueError("Nonfinite SADF scales")
            sources[name] = {
                "module": prefix.rstrip("."), "channels": int(settings[1]), "dim": int(settings[2]),
                "offset_limit_p3_pixels": float(settings[3]), "maximum_absolute_channel_scale": float(settings[4]),
                "alignment": bool(settings[5]), "detail": bool(settings[6]), "scale_selection": bool(settings[7]),
                "checkpoint_recompute": bool(settings[8]),
                "channel_scale_mean": alpha.mean().item(), "channel_scale_std": alpha.std(unbiased=False).item(),
                "channel_scale_min": alpha.min().item(), "channel_scale_max": alpha.max().item(),
                "channel_scales": alpha.tolist(),
                "restore_weight_l2": state[prefix + "restore.weight"].float().norm().item(),
                "p4_offset_predictor_weight_l2": state[prefix + "alignments.0.offset.predict.weight"].float().norm().item(),
                "p5_offset_predictor_weight_l2": state[prefix + "alignments.1.offset.predict.weight"].float().norm().item(),
            }
    if not sources:
        raise ValueError("No SADF model/EMA state found; do not supply eval.pth or a baseline checkpoint")
    return {"schema_version": 1, "checkpoint": str(Path(path).resolve()),
            "last_epoch": checkpoint.get("last_epoch"), "sources": sources,
            "note": "Parameter values do not establish mechanism effectiveness or anatomical attention. "
                    "Best checkpoint is not necessarily final epoch. Offset weight norms are NOT actual displacement magnitudes."}


def export_checkpoint(checkpoint_path, output=None):
    path = Path(checkpoint_path).resolve()
    output = Path(output).resolve() if output else path.with_name(path.stem + "_SADF_parameters.json")
    if output.suffix != ".json" or output == path:
        raise ValueError("Output must be a separate JSON, never the checkpoint")
    if output.exists():
        raise FileExistsError(f"Preserving existing report: {output}")
    report = checkpoint_report(torch.load(path, map_location="cpu", weights_only=True), path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for source, values in report["sources"].items():
        print(f"{source} SADF: channel scale mean={values['channel_scale_mean']:.8f}, "
              f"range=[{values['channel_scale_min']:.8f},{values['channel_scale_max']:.8f}], "
              f"offset bound={values['offset_limit_p3_pixels']:.2f} P3 pixels", flush=True)
    print(f"Saved: {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
