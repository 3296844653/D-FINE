"""Export learned AQS scalars from an existing checkpoint, without inference.

Requires only PyTorch, not the dataset, a GPU, or the training environment's
full D-FINE registration imports. Does not modify the checkpoint.
"""

import argparse
import importlib.util
from pathlib import Path

import torch


def _load_helper():
    path = Path(__file__).resolve().parents[2] / "src/solver/aqs_parameters.py"
    spec = importlib.util.spec_from_file_location("dfine_aqs_parameter_export", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def export_checkpoint(checkpoint_path, output_path=None, temperature=None, overwrite=False):
    helper = _load_helper()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    output_path = (Path(output_path).resolve() if output_path is not None else
                   checkpoint_path.with_name(checkpoint_path.stem + "_aqs_parameters.json"))
    if output_path.suffix.lower() != ".json":
        raise ValueError("--output must be a .json file, never a checkpoint path")
    if output_path == checkpoint_path:
        raise ValueError("Do not overwrite the model checkpoint")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite to replace this JSON")
    # No automatic fallback to arbitrary pickle execution.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    report = helper.checkpoint_aqs_report(checkpoint, checkpoint_path, temperature=temperature)
    helper.save_aqs_report(report, output_path)
    helper.print_aqs_report(report)
    print(f"Saved: {output_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: <checkpoint_stem>_aqs_parameters.json beside checkpoint")
    parser.add_argument("--temperature", type=float,
                        help="Optional original-run fixed temperature; NOT recovered from checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSON only")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.temperature, args.overwrite)


if __name__ == "__main__":
    main()
