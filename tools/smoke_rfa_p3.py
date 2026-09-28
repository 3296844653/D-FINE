"""Check the optional EduYOLO-inspired P3 refinement before a full run.

Run from the repository root: python tools/smoke_rfa_p3.py
Requires PyTorch and the project's ordinary training dependencies, but no data.
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
    baseline = YAMLConfig(
        "configs/dfine/dfine_s_university_5cls.yml",
        HGNetv2={"pretrained": False},
    ).model.to(device).eval()
    trial_cfg = YAMLConfig(
        "configs/dfine/dfine_s_university_5cls_rfa_p3.yml",
        HGNetv2={"pretrained": False},
    )
    trial = trial_cfg.model.to(device).eval()
    disabled = YAMLConfig(
        "configs/dfine/dfine_s_university_5cls_rfa_p3.yml",
        HGNetv2={"pretrained": False},
        HybridEncoder={"use_rfa_p3": False},
    ).model.to(device).eval()

    result = trial.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    assert result.missing_keys and all(
        key.startswith("encoder.rfa_p3.") for key in result.missing_keys
    ), result.missing_keys
    disabled.load_state_dict(baseline.state_dict(), strict=True)

    image = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        base_out = baseline(image)
        trial_out = trial(image)
        disabled_out = disabled(image)
    for key in ("pred_logits", "pred_boxes"):
        assert base_out[key].shape == trial_out[key].shape
        assert torch.allclose(base_out[key], trial_out[key], rtol=1e-6, atol=1e-6), key
        assert torch.allclose(base_out[key], disabled_out[key], rtol=1e-6, atol=1e-6), key

    targets = [{
        "labels": torch.tensor([0, 1, 2], dtype=torch.long, device=device),
        "boxes": torch.tensor(
            [[0.25, 0.30, 0.12, 0.20], [0.52, 0.50, 0.16, 0.25], [0.77, 0.66, 0.13, 0.18]],
            device=device,
        ),
    }]
    trial.train()
    with torch.no_grad():
        train_out = trial(image, targets)
        losses = trial_cfg.criterion.to(device)(train_out, targets)
    assert "aux_outputs" in train_out and "dn_outputs" in train_out
    assert train_out["pred_logits"].shape == trial_out["pred_logits"].shape
    assert losses and all(torch.isfinite(loss).all() for loss in losses.values())

    # Zero initialization makes the first prediction identical, but the new
    # branch still receives gradients and can learn during ordinary training.
    head = trial.encoder.rfa_p3
    sample = torch.randn(1, trial.encoder.hidden_dim, 16, 16, device=device)
    head(sample).sum().backward()
    grad = head.expand.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    extra = sum(p.numel() for p in head.parameters())
    print(f"PASS: baseline-equivalent initialization and training forward; extra parameters={extra}")
    print(f"Prediction shapes: {tuple(trial_out['pred_logits'].shape)}, {tuple(trial_out['pred_boxes'].shape)}")


if __name__ == "__main__":
    main()
