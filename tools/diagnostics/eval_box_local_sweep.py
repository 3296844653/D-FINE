"""Test-only BoxLocal ON/OFF; retain strict state/settings and unchanged original losses."""
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def evaluate(mode, args, runner=runpy.run_path):
    if (mode not in ('on','off') or '--test-only' not in args
            or not any(x in args for x in ('-r','--resume')) or any(x in args for x in ('-t','--tuning'))
            or '-u' not in args or f'diagnostic_box_local_output_enabled={str(mode=="on").lower()}' not in args
            or f'diagnostic_box_local_ablation=box_{mode}' not in args):
        raise ValueError('Existing-checkpoint test-only ON/OFF and actual runtime metadata are required')
    from src.zoo.dfine.night_probes import BoxLocalQueryAdapter
    original, argv = BoxLocalQueryAdapter.__init__,sys.argv
    def constructor(self,*a,**kw):
        original(self,*a,**kw);self.enabled = mode == 'on'
    try:
        BoxLocalQueryAdapter.__init__ = constructor
        sys.argv = [str(ROOT/'train.py'),*args]
        return runner(str(ROOT/'train.py'),run_name='__main__')
    finally:
        BoxLocalQueryAdapter.__init__,sys.argv = original,argv


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--mode',required=True,choices=('on','off'));a,rest = p.parse_known_args()
    evaluate(a.mode,rest)


if __name__ == '__main__':main()
