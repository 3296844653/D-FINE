"""Server-side checks for Exp-1; no checkpoint, training or dataset required.

Run from the repository root:
    python tools/check_single_label_postprocessor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.core import YAMLConfig
from src.zoo.dfine.postprocessor import DFINEPostProcessor


def check_outputs(device, dtype):
    # Both highest original class scores belong to query 0.
    logits = torch.tensor([[[5., 4.], [3., -4.], [-2., 2.]]], device=device, dtype=dtype)
    boxes = torch.tensor([[[.2, .3, .1, .2], [.5, .6, .2, .2], [.8, .8, .1, .1]]],
                         device=device, dtype=dtype)
    sizes = torch.tensor([[100, 200]], device=device)
    outputs = {'pred_logits': logits, 'pred_boxes': boxes}
    original = {key: value.clone() for key, value in outputs.items()}
    old = DFINEPostProcessor(num_classes=2, num_top_queries=2)
    off = DFINEPostProcessor(num_classes=2, num_top_queries=2, single_label_per_query=False)
    on = DFINEPostProcessor(num_classes=2, num_top_queries=2, single_label_per_query=True)
    old_result, off_result, new_result = old(outputs, sizes)[0], off(outputs, sizes)[0], on(outputs, sizes)[0]
    for key in old_result:
        assert torch.equal(old_result[key], off_result[key]), key
    assert old_result['labels'].tolist() == [0, 1]
    assert torch.equal(old_result['boxes'][0], old_result['boxes'][1])
    assert new_result['labels'].tolist() == [0, 0]
    torch.testing.assert_close(new_result['scores'], logits.sigmoid()[0, [0, 1], [0, 0]])
    # Explicit box conversion confirms sorted scores select the correct queries.
    expected = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                          boxes[..., :2] + boxes[..., 2:] / 2), -1) * sizes.repeat(1, 2).unsqueeze(1)
    torch.testing.assert_close(new_result['boxes'], expected[0, :2])
    for key in original:
        assert torch.equal(outputs[key], original[key]), 'Input was modified'
    labels, out_boxes, scores = on.deploy()(outputs, sizes)
    for actual, wanted in zip((labels, out_boxes, scores),
                              (new_result['labels'], new_result['boxes'], new_result['scores'])):
        assert torch.equal(actual[0], wanted)
    assert len(on.state_dict()) == len(off.state_dict()) == 0
    # More requested outputs than available queries must not duplicate queries.
    capped = DFINEPostProcessor(num_classes=2, num_top_queries=10, single_label_per_query=True)
    assert capped(outputs, sizes)[0]['scores'].numel() == 3
    print(f'PASS: golden example, box alignment, disabled flag, deploy, input preservation ({device}, {dtype})')


def check_real_shapes():
    torch.manual_seed(0)
    logits = torch.randn(2, 300, 5)
    boxes = torch.rand(2, 300, 4)
    sizes = torch.tensor([[640, 640], [1920, 1080]])
    outputs = {'pred_logits': logits, 'pred_boxes': boxes}
    pp = DFINEPostProcessor(num_classes=5, num_top_queries=300)
    results = pp(outputs, sizes)
    # Reference is the pre-change flattened top-k algorithm.
    old_scores, flat = logits.sigmoid().flatten(1).topk(300, dim=-1)
    old_labels, query_ids = flat % 5, flat // 5
    converted = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2), -1) * sizes.repeat(1, 2).unsqueeze(1)
    old_boxes = converted.gather(1, query_ids.unsqueeze(-1).expand(-1, -1, 4))
    for batch in range(2):
        assert torch.equal(results[batch]['scores'], old_scores[batch])
        assert torch.equal(results[batch]['labels'], old_labels[batch])
        torch.testing.assert_close(results[batch]['boxes'], old_boxes[batch])
    pp.single_label_per_query = True
    results = pp(outputs, sizes)
    best_scores, best_labels = logits.sigmoid().max(-1)
    order = best_scores.argsort(dim=-1, descending=True)
    for batch in range(2):
        assert results[batch]['boxes'].shape == (300, 4)
        assert torch.equal(results[batch]['labels'], best_labels[batch, order[batch]])
        torch.testing.assert_close(results[batch]['boxes'], converted[batch, order[batch]])
    print('PASS: B=2, Q=300, C=5; original flattened top-k reference and enabled query/label alignment')


def check_config():
    root = Path(__file__).resolve().parents[1]
    base = YAMLConfig(str(root / 'configs/dfine/dfine_s_university_5cls_diagnostics.yml'))
    # Build baseline first: registry configuration is shared in some versions.
    baseline_pp = base.postprocessor
    assert baseline_pp.single_label_per_query is False
    exp = YAMLConfig(str(root / 'configs/dfine/dfine_s_university_5cls_single_label_diagnostics.yml'))
    experiment_pp = exp.postprocessor
    assert experiment_pp.single_label_per_query is True
    assert experiment_pp.num_classes == 5
    assert experiment_pp.num_top_queries == 300
    assert experiment_pp.use_focal_loss is True
    print('PASS: YAML -> registry -> postprocessor; baseline OFF, experiment ON')


if __name__ == '__main__':
    with torch.no_grad():
        check_outputs('cpu', torch.float32)
        check_real_shapes()
        if torch.cuda.is_available():
            check_outputs('cuda', torch.float32)
            check_outputs('cuda', torch.float16)
        check_config()
    print('ALL CHECKS PASSED. No model training or dataset evaluation was performed.')
