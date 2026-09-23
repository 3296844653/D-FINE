"""Server-side smoke check for the two independent D-FINE-S experiments.

Run before training: python tools/check_experiment_2_3.py --config <experiment.yml>
Requires the server's training environment and pretrained backbone files.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Use the server CUDA environment for this smoke check')
    torch.manual_seed(0)
    cfg = YAMLConfig(args.config)
    model = cfg.model.cuda()
    criterion = cfg.criterion.cuda()
    decoder = model.decoder
    region = decoder.use_region_bilinear_cls
    duplicate = decoder.use_duplicate_relation
    assert region != duplicate, 'Select exactly one experiment'
    assert cfg.postprocessor.single_label_per_query is False
    images = torch.randn(1, 3, 640, 640, device='cuda')
    targets = [{
        'labels': torch.tensor([0, 2], device='cuda', dtype=torch.int64),
        'boxes': torch.tensor([[.3, .3, .15, .2], [.7, .65, .12, .18]], device='cuda'),
    }]
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        outputs = model(images, targets)
        losses = criterion(outputs, targets)
        total = sum(losses.values())
    assert outputs['pred_logits'].shape == (1, decoder.num_queries, 5)
    assert outputs['pred_boxes'].shape == (1, decoder.num_queries, 4)
    assert torch.isfinite(total), 'Non-finite training loss'
    if duplicate:
        assert outputs['duplicate_keep_logits'].shape == (1, decoder.num_queries, 1)
        assert torch.isfinite(losses['loss_duplicate_keep'])
    else:
        assert 'duplicate_keep_logits' not in outputs
    total.backward()
    branch = decoder.duplicate_relation if duplicate else decoder.region_bilinear_cls
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters()), 'No branch gradient'
    del outputs, losses, total
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        outputs = model(images)
        results = cfg.postprocessor(outputs, torch.tensor([[640, 640]], device='cuda'))
    assert outputs['pred_logits'].shape == (1, decoder.num_queries, 5)
    assert len(results) == 1 and results[0]['boxes'].shape[-1] == 4
    assert torch.isfinite(outputs['pred_logits']).all()
    print('PASS: train/criterion/backward/eval/postprocessor with AMP, DN and auxiliary losses')
    print('Experiment:', 'duplicate_relation' if duplicate else 'region_bilinear_cls')


if __name__ == '__main__':
    main()
