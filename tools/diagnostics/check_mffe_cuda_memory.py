"""Two real-data CUDA/AMP optimization steps; never save or resume weights.

Checks the configured batch at the base and maximum training resolutions.
This is a memory smoke test, not a training result or a full-run guarantee.
"""

import argparse
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/dfine_s_scbs_hrw_mffe.yml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required: run this preflight on the 4090 server")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = YAMLConfig(str(config_path))
    if not cfg.yaml_cfg.get("HybridEncoder", {}).get("use_mffe", False):
        parser.error("The config must enable MFFE")
    # Architecture and CUDA memory do not require pretrained downloads. The
    # randomly initialized smoke model is discarded, never used for metrics.
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # Collate runs in the main process so each requested size is deterministic;
    # do not change the configured batch or actual dataset/augmentations.
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    if not all(m.use_checkpoint for m in model.encoder.mffe):
        parser.error("Enable mffe_checkpoint before running the CUDA preflight")
    optimizer = cfg.optimizer
    # Random untrained weights can overflow the default initial loss scale.
    # A smaller scale avoids false alarms in this memory-only smoke; the
    # actual training configuration/scaler is never changed.
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    ema = cfg.ema
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    collate = loader.collate_fn
    sizes = sorted({int(collate.base_size), max(collate.scales or [collate.base_size])})
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}; capacity={props.total_memory / 2**30:.2f} GiB; "
          f"batch={loader.batch_size}; sizes={sizes}", flush=True)
    print("Random-weight real-data smoke only; no checkpoints/logs will be saved.", flush=True)
    iterator = iter(loader)
    for index, size in enumerate(sizes):
        collate.scales = [size]
        # The iterator has no worker prefetch; collate sees the selected size.
        samples, targets = next(iterator)
        if samples.shape[-2:] != (size, size) or samples.shape[0] != loader.batch_size:
            raise RuntimeError(f"Unexpected smoke batch: {tuple(samples.shape)}")
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in target.items()} for target in targets]
        with torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=True):
            outputs = model(samples, targets=targets)
        # Match det_engine.py: criterion is evaluated outside autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(outputs, targets, epoch=0, step=index, global_step=index,
                               epoch_step=len(loader))
        loss = sum(losses.values())
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError("Non-finite smoke loss")
        scaler.scale(loss).backward()
        if any(p.grad is None for p in model.encoder.mffe.parameters()):
            raise RuntimeError("MFFE parameter did not receive a gradient")
        scaler.unscale_(optimizer)
        if any(not torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None):
            raise RuntimeError("Non-finite AMP gradients during the preflight")
        # One real optimization step includes AdamW's lazy state allocation.
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        torch.cuda.synchronize(device)
        free, _ = torch.cuda.mem_get_info(device)
        print(f"PASS {size}x{size}: max_GT_in_batch="
              f"{max(len(t['labels']) for t in targets)}; "
              f"peak_allocated={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB; "
              f"peak_reserved={torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB; "
              f"free_after_step={free / 2**30:.2f} GiB", flush=True)
        optimizer.zero_grad(set_to_none=True)
        del samples, targets, outputs, losses, loss
    print("Preflight passed for these batches. This does not guarantee every "
          "batch fits; denser targets, CUDA workspaces and other GPU processes "
          "can change the peak. No experiment weights were saved.", flush=True)


if __name__ == "__main__":
    main()
