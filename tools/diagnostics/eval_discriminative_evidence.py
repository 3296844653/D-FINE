"""Evaluation-only ON/OFF entry point; never changes the checkpoint schema."""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def evaluate(mode, train_args, *, module_class=None, runner=runpy.run_path):
    if mode not in ("on", "off"):
        raise ValueError("Choose on or off")
    if "--test-only" not in train_args or not any(k in train_args for k in ("-r", "--resume")):
        raise ValueError("This entry point requires --test-only and an existing checkpoint via -r")
    if any(k in train_args for k in ("-t", "--tuning")):
        raise ValueError("Evaluation must not tune/train a detector")
    flag = f"diagnostic_DiscriminativeEvidenceL2_output_enabled={'true' if mode == 'on' else 'false'}"
    if "-u" not in train_args or flag not in train_args:
        raise ValueError("ON/OFF metadata flag is absent or disagrees with the actual runtime mode")
    if module_class is None:
        from src.zoo.dfine.discriminative_evidence import DiscriminativeEvidenceL2
        module_class = DiscriminativeEvidenceL2
    original = module_class.forward
    original_argv = sys.argv
    try:
        if mode == "off":
            def disabled(self, *args, **kwargs):
                # The model is still constructed and strict-loaded with every
                # trained parameter. Only the runtime residual is bypassed.
                self.enabled = False
                return original(self, *args, **kwargs)
            module_class.forward = disabled
        sys.argv = [str(ROOT/"train.py"), *train_args]
        print(f"Evaluation ONLY: DiscriminativeEvidenceL2={mode}; AQS remains ON", flush=True)
        return runner(str(ROOT/"train.py"), run_name="__main__")
    finally:
        module_class.forward = original
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("on", "off"), required=True)
    args, train_args = parser.parse_known_args()
    evaluate(args.mode, train_args)


if __name__ == "__main__":
    main()
