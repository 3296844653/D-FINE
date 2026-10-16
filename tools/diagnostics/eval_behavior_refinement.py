"""Strict-schema test-only ON/OFF/within-module ablation; keeps AQS enabled."""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def evaluate(mode,train_args,base_class=None,runner=runpy.run_path):
    if mode not in ('on','off','no_memory','no_contrast','no_geometry','no_quality'):
        raise ValueError('Unknown runtime ablation')
    if ('--test-only' not in train_args or not any(x in train_args for x in ('-r','--resume'))
            or any(x in train_args for x in ('-t','--tuning'))):
        raise ValueError('Existing-checkpoint test-only evaluation required')
    enabled = mode != 'off'
    flag = f'diagnostic_behavior_refinement_output_enabled={str(enabled).lower()}'
    if '-u' not in train_args or flag not in train_args or f'diagnostic_behavior_refinement_ablation={mode}' not in train_args:
        raise ValueError('Actual ON/OFF/ablation metadata flags are required')
    if base_class is None:
        from src.zoo.dfine.behavior_refinement import BehaviorRefinementBase
        base_class = BehaviorRefinementBase
    original,argv = base_class.__init__,sys.argv
    def constructor(self,*args,**kwargs):
        original(self,*args,**kwargs)
        allowed = {'ClassEvidenceTransport':{'no_memory','no_contrast','no_quality'},
                   'QueryContrastInitialization':{'no_contrast'},
                   'GeometryAwareQueryCompetition':{'no_geometry','no_contrast','no_quality'},
                   'PrototypeBoundaryRefinement':{'no_contrast','no_quality'}}
        if mode not in ('on','off') and mode not in allowed[self.name]:
            raise ValueError('Ablation is not a component of this architecture')
        self.enabled = enabled
        if mode.startswith('no_'): setattr(self,mode[3:]+'_enabled',False)
    try:
        base_class.__init__ = constructor
        sys.argv = [str(ROOT/'train.py'),*train_args]
        print(f'Evaluation only: new behavior module={mode}, AQS remains ON',flush=True)
        return runner(str(ROOT/'train.py'),run_name='__main__')
    finally:
        base_class.__init__ = original;sys.argv = argv


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--mode',required=True,choices=('on','off','no_memory','no_contrast','no_geometry','no_quality'))
    args,train_args = parser.parse_known_args()
    evaluate(args.mode,train_args)


if __name__=='__main__': main()
