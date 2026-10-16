"""Strict-checkpoint test-only SERF ablations; AQS remains enabled."""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODES = ('on', 'off', 'no_geometry', 'no_relation')


def evaluate(mode, train_args, runner=runpy.run_path):
    if (mode not in MODES or '--test-only' not in train_args
            or not any(x in train_args for x in ('-r', '--resume'))
            or any(x in train_args for x in ('-t', '--tuning'))):
        raise ValueError('SERF ablation requires existing-checkpoint test-only evaluation')
    flag = f'diagnostic_serf_output_enabled={str(mode != "off").lower()}'
    # YAML interprets bare `on`/`off` as booleans; prefix preserves a string.
    if '-u' not in train_args or flag not in train_args or f'diagnostic_serf_ablation=serf_{mode}' not in train_args:
        raise ValueError('Actual SERF ablation metadata flags required')
    from src.zoo.dfine.spatial_evidence_relation_fusion import SpatialEvidenceRelationFusion
    original, argv = SpatialEvidenceRelationFusion.__init__, sys.argv
    def constructor(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.enabled = mode != 'off'
        self.geometry_enabled = mode != 'no_geometry'
        self.relation_enabled = mode != 'no_relation'
    try:
        SpatialEvidenceRelationFusion.__init__ = constructor
        sys.argv = [str(ROOT / 'train.py'), *train_args]
        print(f'Evaluation only: SERF={mode}, AQS remains ON', flush=True)
        return runner(str(ROOT / 'train.py'), run_name='__main__')
    finally:
        SpatialEvidenceRelationFusion.__init__, sys.argv = original, argv


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', required=True, choices=MODES)
    args, train_args = parser.parse_known_args()
    evaluate(args.mode, train_args)


if __name__ == '__main__':
    main()
