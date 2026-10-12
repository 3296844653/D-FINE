"""Export actual EvidenceContrastAQS settings/scalars from a trusted own checkpoint."""
import argparse
import json
from pathlib import Path

import torch


def checkpoint_report(checkpoint, path):
    sources = {}
    for source in ("model", "ema"):
        state = checkpoint.get(source)
        if not isinstance(state, dict):
            continue
        state = state.get("module", state)
        modules = {}
        for key, settings in state.items():
            if not key.endswith(".settings") or "query_cls_refiner" not in key and "aqs_refiners" not in key:
                continue
            settings = settings.detach().float().cpu()
            if settings.numel() != 11 or int(settings[0]) != 1:
                raise ValueError("Unsupported EvidenceContrastAQS settings schema")
            prefix = key.removesuffix("settings")
            tau = state[prefix+"threshold_logit"].detach().float().cpu()
            rho = state[prefix+"residual_scale"].detach().float().cpu()
            if not torch.isfinite(torch.cat((settings,tau.reshape(1),rho.reshape(1)))).all():
                raise ValueError("Nonfinite module parameters")
            modules[prefix.rstrip(".")] = {
                "decoder_layer": int(state[prefix+"applied_layer"]),
                "hidden_dim": int(settings[1]), "classes": int(settings[2]), "dim": int(settings[3]),
                "slots_per_class": int(settings[4]), "heads": int(settings[5]), "temperature": float(settings[6]),
                "geometry": bool(settings[7]), "quality_proxy": bool(settings[8]),
                "support_alternative_contrast": bool(settings[9]), "adaptive_receiver": bool(settings[10]),
                "threshold": tau.sigmoid().item(), "threshold_logit": tau.item(),
                "residual_scale": rho.item(), "effective_residual_scale": rho.tanh().item(),
                "output_projection_weight_l2": state[prefix+"fuse.2.weight"].float().norm().item(),
                "slot_assignment_weight_l2": state[prefix+"slot_assign.weight"].float().norm().item(),
                "geometry_bias_weight_l2": state[prefix+"geometry_bias.2.weight"].float().norm().item(),
                "receiver_weight_l2": state[prefix+"receiver.2.weight"].float().norm().item(),
            }
        if modules:
            sources[source] = modules
    if not sources:
        raise ValueError("No EvidenceContrastAQS state; do not use a baseline/legacy AQS/eval.pth checkpoint")
    return dict(schema_version=1, checkpoint=str(Path(path).resolve()),
        last_epoch=checkpoint.get("last_epoch"), sources=sources,
        note="Best checkpoint need not be final epoch. Scalars/weight norms alone do not prove "
             "correct evidence, confusion reduction or attention to hands. Runtime evidence maps require forward inputs.")


def export_checkpoint(path, output=None):
    path = Path(path).resolve()
    output = Path(output).resolve() if output else path.with_name(path.stem+"_EvidenceContrastAQS_parameters.json")
    if output.suffix != ".json" or output == path:
        raise ValueError("Output must be a separate JSON")
    if output.exists():
        raise FileExistsError(f"Preserving existing report: {output}")
    report = checkpoint_report(torch.load(path, map_location="cpu", weights_only=True),path)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    for source,modules in report['sources'].items():
        for name,values in modules.items():
            print(f"{source} {name} L{values['decoder_layer']}: threshold={values['threshold']:.8f}, "
                  f"effective residual={values['effective_residual_scale']:.8f}, "
                  f"T={values['temperature']:.4f}, slots={values['slots_per_class']}",flush=True)
    print(f"Saved: {output}",flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint,args.output)
