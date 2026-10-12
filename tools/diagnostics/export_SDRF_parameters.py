"""Export SDRF model/EMA parameters from a trusted own training checkpoint."""
import argparse
import json
from pathlib import Path

import torch


def checkpoint_report(checkpoint, path):
    sources = {}
    for name in ('model', 'ema'):
        state = checkpoint.get(name)
        if not isinstance(state, dict):
            continue
        state = state.get('module', state)
        for key, raw in state.items():
            if not key.endswith('encoder.sdrf.raw_alpha'):
                continue
            prefix = key.removesuffix('raw_alpha')
            settings = state[prefix + 'settings'].detach().cpu()
            if settings.dtype != torch.int64 or settings.numel() != 9 or settings[0].item() != 1:
                raise ValueError('Unsupported SDRF checkpoint settings')
            strength = state[prefix + 'semantic_strength_logit'].detach().float().cpu().sigmoid()
            alpha = .5 * raw.detach().float().cpu().tanh()
            weight_norm = state[prefix + 'fuse.output.weight'].detach().float().cpu().norm()
            if not all(torch.isfinite(v).all() for v in (strength, alpha, weight_norm)):
                raise ValueError('Nonfinite SDRF parameters')
            sources[name] = dict(module=prefix.rstrip('.'), channels=int(settings[1]),
                dim=int(settings[2]), heads=int(settings[3]), window=int(settings[4]),
                halo=int(settings[5]), checkpoint_recompute=bool(settings[6]),
                semantic_condition=bool(settings[7]), differential=bool(settings[8]),
                effective_residual_scale=alpha.item(), maximum_absolute_scale=.5,
                semantic_strength_per_head=strength.tolist(), output_weight_l2=weight_norm.item())
    if not sources:
        raise ValueError('No SDRF state: supply its training checkpoint, not baseline/eval.pth')
    return dict(schema_version=1, checkpoint=str(Path(path).resolve()),
        last_epoch=checkpoint.get('last_epoch'), sources=sources,
        note='Learned parameters alone do not prove reduced confusion, noise cancellation or anatomical attention. '
             'A best checkpoint need not be the final epoch. Report EMA for EMA evaluation.')


def export_checkpoint(checkpoint_path, output=None):
    path = Path(checkpoint_path).resolve()
    destination = Path(output).resolve() if output else path.with_name(path.stem + '_SDRF_parameters.json')
    if destination.suffix != '.json' or destination == path:
        raise ValueError('Output must be a separate JSON')
    if destination.exists():
        raise FileExistsError(f'Preserving existing report: {destination}')
    report = checkpoint_report(torch.load(path, map_location='cpu', weights_only=True), path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for source, values in report['sources'].items():
        print(f"{source} SDRF: scale={values['effective_residual_scale']:.8f}, "
              f"semantic_strength={values['semantic_strength_per_head']}", flush=True)
    print('Saved:', destination, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
