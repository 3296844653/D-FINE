"""Evaluation-only SCF residual ON/OFF, retaining AQS and checkpoint schema."""
import argparse
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--module-mode",choices=("on","off"),required=True)
    args,remaining=parser.parse_known_args()
    if "--test-only" not in remaining or not any(s in remaining for s in ("-r","--resume")):
        parser.error("ON/OFF wrapper requires checkpoint evaluation: -r and --test-only")
    if any(s in remaining for s in ("-t","--tuning")):
        parser.error("ON/OFF wrapper must not tune/train")
    marker=f"diagnostic_SubpixelContextFusion_output_enabled={'true' if args.module_mode == 'on' else 'false'}"
    if marker not in remaining:
        parser.error("Missing matching diagnostic ON/OFF metadata flag")
    from src.zoo.dfine.subpixel_context_fusion import SubpixelContextFusion
    original=SubpixelContextFusion.forward
    argv=sys.argv[:]
    def toggled(module,*pos,**kw):
        previous=module.enabled
        try:
            module.enabled=args.module_mode == "on"
            return original(module,*pos,**kw)
        finally:
            module.enabled=previous
    try:
        SubpixelContextFusion.forward=toggled
        sys.argv=[str(ROOT/"train.py"),*remaining]
        print("SCF runtime:",args.module_mode,"; AQS remains ON",flush=True)
        runpy.run_path(str(ROOT/"train.py"),run_name="__main__")
    finally:
        SubpixelContextFusion.forward=original
        sys.argv=argv


if __name__ == "__main__": main()
