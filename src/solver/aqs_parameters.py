"""Read-only AQS scalar snapshots; does not run a model or change its state."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch


def _is_aqs_path(name):
    parts = name.split(".")
    if parts[-1] == "query_cls_refiner":
        return True
    return _layer_from_path(name) is not None


def _layer_from_path(name):
    parts = name.split(".")
    if len(parts) < 2 or parts[-2] != "aqs_refiners":
        return None
    layer = parts[-1].removeprefix("layer")
    if parts[-1].startswith("layer") and layer.isdigit() and int(layer) > 0:
        return int(layer)
    return None


def _values(threshold_logit, residual_scale, temperature=None, temperature_source=None):
    values = []
    for name, tensor in (("threshold_logit", threshold_logit),
                         ("residual_scale", residual_scale)):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
            raise ValueError(f"AQS {name} must be a scalar tensor")
        value = tensor.detach().cpu().float()
        if not torch.isfinite(value).all():
            raise ValueError(f"AQS {name} is not finite")
        values.append(value)
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("AQS fixed temperature must be finite and positive")
    threshold, residual = values
    return {
        "threshold_logit": threshold.item(),
        "threshold": threshold.sigmoid().item(),
        "residual_scale": residual.item(),
        "effective_residual_scale": residual.tanh().item(),
        "temperature": temperature,
        "temperature_source": temperature_source,
    }


def runtime_aqs_report(model, ema=None, *, epoch, snapshot="evaluated_epoch"):
    """Capture model and EMA independently, before any solver stage-2 reload.

    Only detach each refiner's two scalars. Do not copy the full GPU state_dict, consume
    RNG, change train/eval mode, or write on distributed non-master workers.
    The caller is responsible for the main-process check.
    """
    sources = {}
    modules = {"model": model}
    if ema is not None:
        modules["ema"] = ema.module
    for source, model_module in modules.items():
        if model_module is None:
            continue
        parameters = {}
        for name, module in model_module.named_modules():
            if not _is_aqs_path(name):
                continue
            if not all(hasattr(module, attr) for attr in
                       ("threshold_logit", "residual_scale", "temperature")):
                continue
            # DataParallel/DDP's wrapper prefix is not part of the model name.
            name = name.removeprefix("module.")
            parameters[name] = _values(
                module.threshold_logit, module.residual_scale, module.temperature,
                "runtime_fixed_not_learned",
            )
            if hasattr(module, "applied_decoder_layer"):
                parameters[name]["decoder_layer"] = module.applied_decoder_layer
        if parameters:
            sources[source] = parameters
    if not sources:
        return None
    return {
        "schema_version": 1,
        "snapshot": snapshot,
        "epoch": int(epoch),
        "epoch_numbering": "zero-based training loop epoch, not checkpoint last_epoch",
        "evaluation_weight_source": "ema" if ema is not None else "model",
        "sources": sources,
    }


def checkpoint_aqs_report(checkpoint, path, *, temperature=None):
    """Extract both sources from a loaded trusted-format checkpoint.

    Temperature is a plain Python float, NOT a state_dict tensor. Never infer
    its value from this source's default or from another experiment's YAML.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a model checkpoint dictionary")
    sources = {}
    for source in ("model", "ema"):
        state = checkpoint.get(source)
        if not isinstance(state, Mapping):
            continue
        if isinstance(state.get("module"), Mapping):
            state = state["module"]
        parameters = {}
        for key, value in state.items():
            if not isinstance(key, str) or not key.endswith(".threshold_logit"):
                continue
            prefix = key.removesuffix(".threshold_logit")
            if not _is_aqs_path(prefix):
                continue
            residual_key = prefix + ".residual_scale"
            if residual_key not in state:
                raise ValueError(f"Incomplete AQS checkpoint: missing {residual_key}")
            name = prefix.removeprefix("module.")
            parameters[name] = _values(
                value, state[residual_key], temperature,
                "user_supplied_not_stored_in_checkpoint" if temperature is not None else None,
            )
            layer = _layer_from_path(name)
            if layer is not None:
                parameters[name]["decoder_layer"] = layer
        if parameters:
            sources[source] = parameters
    if not sources:
        raise ValueError("No AQS parameters in model/EMA; use the AQS model checkpoint, not eval.pth")
    return {
        "schema_version": 1,
        "snapshot": "checkpoint",
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_last_epoch": checkpoint.get("last_epoch"),
        "note": "best_stg2 is the best saved checkpoint, not necessarily the final epoch; "
                "the original solver's last.pth may stop updating at stage 1. "
                "Fixed temperature is not stored in the checkpoint. "
                "Multi-layer AQS names record their layer numbers; legacy single-layer "
                "checkpoints need their original config to identify the insertion layer.",
        "sources": sources,
    }


def save_aqs_report(report, path, *, snapshot=None, checkpoint=None):
    """Save JSON only; never overwrite or save model checkpoint tensors."""
    if report is None:
        return
    report = dict(report)
    if snapshot is not None:
        report["snapshot"] = snapshot
    if checkpoint is not None:
        report["checkpoint"] = str(checkpoint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def print_aqs_report(report):
    if report is None:
        return
    epoch = report.get("epoch", report.get("checkpoint_last_epoch"))
    print(f"AQS parameters: {report['snapshot']}, epoch={epoch}", flush=True)
    for source, modules in report["sources"].items():
        for name, values in modules.items():
            temp = values["temperature"]
            temp_text = "unknown (not stored in checkpoint)" if temp is None else f"{temp:.8f} (fixed)"
            layer_text = f" [L{values['decoder_layer']}]" if "decoder_layer" in values else ""
            print(
                f"  {source} {name}{layer_text}: threshold={values['threshold']:.8f}, "
                f"threshold_logit={values['threshold_logit']:.8f}, "
                f"residual_scale={values['residual_scale']:.8f}, "
                f"effective_residual_scale={values['effective_residual_scale']:.8f}, "
                f"temperature={temp_text}",
                flush=True,
            )
