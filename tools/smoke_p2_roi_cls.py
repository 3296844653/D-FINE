"""Minimal CPU/GPU smoke check for the optional P2 ROI class refiner.

Run from the repository root: python tools/smoke_p2_roi_cls.py
No dataset or pretrained checkpoint is required.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.core import YAMLConfig


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = YAMLConfig(
        "configs/dfine/dfine_s_university_5cls.yml",
        HGNetv2={"pretrained": False},
    ).model.to(device).eval()
    refined = YAMLConfig(
        "configs/dfine/dfine_s_university_5cls_p2_roi_cls.yml",
        HGNetv2={"pretrained": False},
    ).model.to(device).eval()
    switched_off = YAMLConfig(
        "configs/dfine/dfine_s_university_5cls_p2_roi_cls.yml",
        HGNetv2={"pretrained": False},
        DFINETransformer={"use_p2_roi_cls": False},
    ).model.to(device).eval()

    load_result = refined.load_state_dict(base.state_dict(), strict=False)
    assert not load_result.unexpected_keys, load_result.unexpected_keys
    assert load_result.missing_keys and all(
        name.startswith("decoder.p2_roi_cls.") for name in load_result.missing_keys
    ), load_result.missing_keys
    switched_off.load_state_dict(base.state_dict(), strict=True)

    image = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        base_out = base(image)
        refined_out = refined(image)
        switched_off_out = switched_off(image)
    for key in ("pred_logits", "pred_boxes"):
        assert base_out[key].shape == refined_out[key].shape
        assert torch.allclose(base_out[key], refined_out[key], atol=1e-6, rtol=1e-6), key
        assert torch.allclose(base_out[key], switched_off_out[key], atol=1e-6, rtol=1e-6), key

    refined.train()
    targets = [{
        "labels": torch.tensor([0, 1, 2], dtype=torch.long, device=device),
        "boxes": torch.tensor(
            [[0.25, 0.30, 0.12, 0.20], [0.52, 0.50, 0.16, 0.25], [0.77, 0.66, 0.13, 0.18]],
            device=device,
        ),
    }]
    with torch.no_grad():
        train_out = refined(image, targets)
    assert train_out["pred_logits"].shape == refined_out["pred_logits"].shape
    assert "dn_outputs" in train_out and "aux_outputs" in train_out

    # Verify that the new head can receive a training gradient. Its final
    # linear layer is zero-initialized, so the initial detector is unchanged.
    head = refined.decoder.p2_roi_cls
    feature = torch.randn(1, 64, 160, 160, device=device)
    boxes = torch.rand(1, 4, 4, device=device)
    boxes[..., 2:] *= 0.25
    head(feature, boxes).sum().backward()
    assert head.classifier.weight.grad is not None
    assert torch.isfinite(head.classifier.weight.grad).all()

    extra_params = sum(p.numel() for p in head.parameters())
    print(f"PASS: zero-initialized output matches baseline; extra parameters={extra_params}")
    print(f"Prediction shapes: logits={tuple(refined_out['pred_logits'].shape)}, boxes={tuple(refined_out['pred_boxes'].shape)}")


if __name__ == "__main__":
    main()
