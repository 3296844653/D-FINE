"""Stdlib-only exporter/metadata tests; no real GPU or accuracy claims."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from eval_discriminative_evidence import evaluate
from scbs_discriminative_evidence_export import audit, digest, package, training_log

SCRIPT = ROOT/"tools/diagnostics/export_scbs_AQS_DiscriminativeEvidenceL2.sh"
NAME = "dfine_s_scbs_hrw_epochs110_AQS_DiscriminativeEvidenceL2_seed0_run1"

# A fake Conda is used ONLY inside temporary test directories. It simulates
# external programs without importing torch or accessing the server/GPU.
FAKE_CONDA = r'''#!/usr/bin/env python3
import hashlib,json,os,subprocess,sys
from pathlib import Path
a=sys.argv[1:]
assert a[:3]==['run','--no-capture-output','-n'],a
p=a[4:]
with Path(os.environ['DER_FAKE_CALLS']).open('a') as f: f.write(json.dumps(p)+'\n')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f: json.dump(value,f)
if p[0]=='torchrun':
 assert '--test-only' in p and '-r' in p and '-t' not in p,'training attempted'
 assert '--no-python' in p
 mode=p[p.index('--mode')+1];enabled=(mode=='on')
 assert f'diagnostic_DiscriminativeEvidenceL2_output_enabled={str(enabled).lower()}' in p
 if os.environ.get('DER_FAKE_FAIL_ON')=='1' and enabled: sys.exit(7)
 cp=p[p.index('-r')+1];out=Path(p[p.index('--output-dir')+1])
 cfg={'DFINETransformer':{'use_discriminative_evidence_l2':True,'use_aqs_refine':True,'aqs_apply_layer':2},
      'diagnostic_DiscriminativeEvidenceL2_output_enabled':enabled}
 save(out/'query_diagnostics/metadata.json',{'complete':True,'checkpoint_sha256':digest(cp),'config':cfg})
 save(out/'confusion_matrix/confusion_matrix.json',{'matrix_counts':[[1]]})
 save(out/'validation_metrics.json',{'AP':.5523383035882792 if enabled else .5519,'checkpoint':cp})
elif p[0]=='python':
 q=p[1:]
 if q and q[0]=='-u':q=q[1:]
 if q[0]=='-':
  sys.stdin.read()
  if os.environ.get('DER_FAKE_BUSY')=='1':sys.exit(9)
  print('fake GPU/port free')
 elif q[0].endswith('scbs_discriminative_evidence_export.py'):
  if q[1]=='prepare':
   cp,bundle=Path(q[2]),Path(q[3]);bundle.mkdir();(bundle/'logs').mkdir()
   annotation=cp.parent/'validation_fixture.json'
   save(bundle/'export_inputs.json',{'checkpoint':str(cp),'checkpoint_sha256':digest(cp),
     'validation_annotation':str(annotation),'validation_annotation_sha256':digest(annotation),
     'training_log':{'best':{'epoch':105,'AP':.5523383035882792}}})
  else:sys.exit(subprocess.call([sys.executable,*q]))
 elif q[0].endswith('export_aqs_parameters.py'):
  save(q[q.index('--output')+1],{'mode':'read_only'})
 elif q[0].endswith('scbs_discriminative_evidence_experiment.py'):
  if q[1]=='parameters':save(q[3],{'mode':'read_only'})
  elif q[1]=='on-off':save(Path(q[2])/'DiscriminativeEvidenceL2_on_off.json',{'same_weights':True})
  elif q[1]=='compare':save(Path(q[2])/'diagnostics_eval'/f'compare_{q[4]}.json',{'reference':q[3]})
  else:raise AssertionError(q)
 elif q[0].endswith('analyze_detection_diagnostics.py'):
  source=Path(q[q.index('--input-dir')+1]);out=Path(q[q.index('--output-dir')+1])
  meta=json.loads((source/'query_diagnostics/metadata.json').read_text())
  val=json.loads((source/'validation_metrics.json').read_text())
  save(out/'summary.json',{'checkpoint_sha256':meta['checkpoint_sha256'],'weight_source':'ema',
    'saved_COCO_metrics':{'AP':val['AP']}})
 else:raise AssertionError(q)
else:raise AssertionError(p)
'''


class ExportTests(unittest.TestCase):
    def fixture(self, root):
        training=root/NAME;training.mkdir()
        (training/"best_stg2.pth").write_bytes(b"fake checkpoint; do not alter")
        (training/"log.txt").write_text("training log; do not alter")
        (training/"validation_fixture.json").write_text('{}')
        fake=root/"fake_conda";fake.write_text(FAKE_CONDA);fake.chmod(0o700)
        env=dict(os.environ,SCBS_DER_ROOT=str(root),SCBS_DER_CONDA=str(fake),
                 DER_FAKE_CALLS=str(root/"calls.jsonl"))
        return training,env

    def launch(self, env, *args):
        return subprocess.run(["bash",str(SCRIPT),*args],env=env,capture_output=True,text=True)

    def test_runtime_guard_never_trains_and_restores_method(self):
        class Fake:
            def __init__(self):self.enabled=True;self.weights={'w':3}
            def forward(self):return 'enhanced' if self.enabled else 'unchanged'
        original=Fake.forward;seen=[]
        def runner(path,run_name):
            instance=Fake();seen.append((instance.forward(),dict(instance.weights)))
        for mode in ('off','on'):
            args=['-r','best.pth','--test-only','-u',
                  f'diagnostic_DiscriminativeEvidenceL2_output_enabled={str(mode=="on").lower()}']
            evaluate(mode,args,module_class=Fake,runner=runner)
            self.assertIs(Fake.forward,original)
        self.assertEqual(seen,[('unchanged',{'w':3}),('enhanced',{'w':3})])
        for args in (['-r','best.pth'],['--test-only'],['-r','best.pth','--test-only','-t','other.pth'],
                     ['-r','best.pth','--test-only','-u','diagnostic_DiscriminativeEvidenceL2_output_enabled=false']):
            with self.assertRaises(ValueError):evaluate('on',args,module_class=Fake,runner=runner)

    def test_runtime_restores_method_on_failure(self):
        class Fake:
            def forward(self):return self.enabled
        original=Fake.forward
        def bad(path,run_name):raise RuntimeError('simulated eval failure')
        with self.assertRaises(RuntimeError):
            evaluate('off',['-r','best.pth','--test-only','-u',
                'diagnostic_DiscriminativeEvidenceL2_output_enabled=false'],module_class=Fake,runner=bad)
        self.assertIs(Fake.forward,original)

    def test_log_duplicates_best_and_missing_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'log.txt'
            log.write_text('\n'.join(json.dumps({'epoch':e,'test_coco_eval_bbox':[ap]})
                for e,ap in [(102,.55),(103,.552),(103,.551),(105,.5523383035882792)])+'\npartial')
            report=training_log(log)
            self.assertEqual(report['records'],4);self.assertEqual(report['unique_epochs'],3)
            self.assertEqual(report['duplicate_epochs'],{'103':2})
            self.assertEqual(report['best']['epoch'],105)
            self.assertIsNone(training_log(Path(tmp)/'absent.txt')['best'])

    def test_dry_run_old_results_missing_weights_and_invalid_ids(self):
        subprocess.run(['bash','-n',str(SCRIPT)],check=True)
        subprocess.run(['bash','-n',str(ROOT/'tools/diagnostics/sync_scbs_AQS_DiscriminativeEvidenceL2_diagnostics.sh')],check=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training,env=self.fixture(root)
            before={p.name:digest(p) for p in training.iterdir()}
            result=self.launch(env,'--dry-run','run1','export1')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse((root/'calls.jsonl').exists())
            self.assertEqual(before,{p.name:digest(p) for p in training.iterdir()})
            old=training/'DiscriminativeEvidenceL2_diagnostics_export1';old.mkdir()
            marker=old/'old.txt';marker.write_text('preserve')
            self.assertNotEqual(self.launch(env,'--dry-run').returncode,0)
            self.assertEqual(marker.read_text(),'preserve')
            self.assertNotEqual(self.launch(env,'--dry-run','run999','export1').returncode,0)
            self.assertNotEqual(self.launch(env,'--dry-run','run1','../other').returncode,0)

    def test_busy_gpu_does_not_create_bundle_or_touch_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            training,env=self.fixture(Path(tmp));env['DER_FAKE_BUSY']='1'
            before=digest(training/'best_stg2.pth')
            result=self.launch(env)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((training/'DiscriminativeEvidenceL2_diagnostics_export1').exists())
            self.assertEqual(digest(training/'best_stg2.pth'),before)

    def test_mock_end_to_end_two_test_only_calls_same_cp_reference_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training,env=self.fixture(root)
            originals={p.name:digest(p) for p in training.iterdir()}
            for name,suffix in [('dfine_s_scbs_hrw_epochs110_aqs_layer2_r010_seed0_run1','diagnostics_eval_v2'),
                                ('dfine_s_scbs_hrw_epochs110_seed0_run2','diagnostics_export1')]:
                folder=root/name/suffix
                for file in ('analysis/summary.json','analysis/gt_records.jsonl','evaluation/query_diagnostics/metadata.json'):
                    path=folder/file;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            result=self.launch(env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            bundle=training/'DiscriminativeEvidenceL2_diagnostics_export1'
            calls=[json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            torchruns=[call for call in calls if call[0]=='torchrun']
            self.assertEqual(len(torchruns),2)
            self.assertEqual([call[call.index('--mode')+1] for call in torchruns],['on','off'])
            for call in torchruns:
                self.assertIn('--test-only',call)
                self.assertEqual(call[call.index('-r')+1],str(training/'best_stg2.pth'))
                self.assertNotIn('-t',call)
            for name,expected in originals.items():self.assertEqual(digest(training/name),expected)
            self.assertTrue(json.loads((bundle/'checkpoint_evaluation_audit.json').read_text())['matches_training_best_AP'])
            for label in ('AQS_L2_R1','BaselineRun2'):
                self.assertTrue((bundle/'diagnostics_eval'/f'compare_{label}.json').is_file())
            with zipfile.ZipFile(bundle/(bundle.name+'.zip')) as z:
                self.assertIn('DiscriminativeEvidenceL2_on_off.json',z.namelist())
                self.assertFalse(any(name.endswith('best_stg2.pth') for name in z.namelist()))

    def test_eval_failure_preserves_partial_bundle_and_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training,env=self.fixture(root);env['DER_FAKE_FAIL_ON']='1'
            before=digest(training/'best_stg2.pth')
            result=self.launch(env)
            self.assertNotEqual(result.returncode,0)
            bundle=training/'DiscriminativeEvidenceL2_diagnostics_export1'
            self.assertTrue(bundle.is_dir());self.assertFalse((bundle/(bundle.name+'.zip')).exists())
            self.assertEqual(digest(training/'best_stg2.pth'),before)
            again=self.launch(env);self.assertNotEqual(again.returncode,0)
            self.assertIn('保留旧导出',again.stderr)

    def test_audit_mismatch_is_reported_and_changed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'best.pth';cp.write_bytes(b'unchanged');ann=root/'ann.json';ann.write_text('{}')
            bundle=root/'export';bundle.mkdir()
            snapshot=dict(checkpoint=str(cp),checkpoint_sha256=digest(cp),validation_annotation=str(ann),
                validation_annotation_sha256=digest(ann),training_log={'best':{'epoch':105,'AP':.5523383035882792}})
            (bundle/'export_inputs.json').write_text(json.dumps(snapshot))
            for name,enabled in [('diagnostics_eval',True),('DiscriminativeEvidenceL2_off_eval',False)]:
                folder=bundle/name
                meta=folder/'evaluation/query_diagnostics/metadata.json';meta.parent.mkdir(parents=True)
                meta.write_text(json.dumps({'complete':True,'checkpoint_sha256':digest(cp),
                    'config':{'diagnostic_DiscriminativeEvidenceL2_output_enabled':enabled}}))
                summary=folder/'analysis/summary.json';summary.parent.mkdir(parents=True)
                summary.write_text(json.dumps({'checkpoint_sha256':digest(cp),'weight_source':'ema','saved_COCO_metrics':{'AP':.4}}))
            audit(bundle)
            self.assertFalse(json.loads((bundle/'checkpoint_evaluation_audit.json').read_text())['matches_training_best_AP'])
            cp.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):audit(bundle)

    def test_package_refuses_training_weights_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'export';bundle.mkdir()
            weight=bundle/'best_stg2.pth';weight.write_bytes(b'do not package')
            with self.assertRaises(ValueError):package(bundle)
            # Use a different clean folder, no deletion of existing fixtures.
            clean=root/'clean';clean.mkdir();(clean/'summary.json').write_text('{}')
            package(clean)
            with self.assertRaises(FileExistsError):package(clean)

    def test_sync_manifest_only_changes_diagnostic_tools_and_doc(self):
        manifest=ROOT/'tools/diagnostics/scbs_AQS_DiscriminativeEvidenceL2_diagnostics_sync_files.txt'
        for name in manifest.read_text().splitlines():
            self.assertTrue((ROOT/name).is_file(),name)
            self.assertTrue(name.startswith(('tools/diagnostics/','docs/')),name)
            self.assertFalse(name.startswith(('src/','configs/','dataset/')),name)


if __name__=='__main__':unittest.main(verbosity=2)
