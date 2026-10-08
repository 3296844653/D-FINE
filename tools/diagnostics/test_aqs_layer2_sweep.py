"""Check the sweep's configs and queue with fake launchers, NEVER train a detector."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diagnostics.test_aqs_refine import fresh_config
from src.solver.aqs_parameters import runtime_aqs_report

SCRIPT = ROOT / 'tools/diagnostics/run_scbs_aqs_layer2_sweep.sh'
PROFILES = {
    'R1': ('r010', .5, .1, .1),
    'R2': ('r020', .5, .2, .1),
    'R3': ('r025', .5, .25, .1),
    'TAU1': ('tau045', .45, .05, .1),
    'TAU2': ('tau055', .55, .05, .1),
    'T1': ('temp005', .5, .05, .05),
    'T2': ('temp020', .5, .05, .2),
    'CONTROL': ('control', .5, .05, .1),
}

# Temporary fixtures only: record arguments and create a dummy checkpoint.
# No torchrun/Conda/GPU training program is launched by this fake executable.
FAKE_CONDA = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['run', '--no-capture-output', '-n'], args
command = args[4:]
if command[0] == 'python':
    sys.exit(int(os.environ.get('AQS_TEST_PREFLIGHT_EXIT', '0')))
assert command[0] == 'torchrun', command
output = Path(command[command.index('--output-dir') + 1])
config = command[command.index('-c') + 1]
with Path(os.environ['AQS_TEST_CALLS']).open('a', encoding='utf-8') as f:
    f.write(json.dumps({'simulated': True, 'command': command, 'config': config,
                        'output': str(output), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
                        'unbuffered': os.environ['PYTHONUNBUFFERED']}) + '\n')
print('SIMULATED TRAINING LOG', output.name, flush=True)
fail_tag = os.environ.get('AQS_TEST_FAIL_TAG', '')
if fail_tag and fail_tag in output.name:
    sys.exit(17)
missing_tag = os.environ.get('AQS_TEST_MISSING_TAG', '')
if not missing_tag or missing_tag not in output.name:
    (output / 'best_stg2.pth').write_bytes(b'fake test checkpoint; not model weights')
'''
FAKE_GPU = r'''
import os, sys
if '--query-compute-apps=pid' in sys.argv:
    print(os.environ.get('AQS_TEST_GPU_PIDS', ''))
else:
    print('SIMULATED NVIDIA-SMI')
'''
FAKE_PS = r'''
import os, sys
if os.environ.get('AQS_TEST_PS_DENIED') == '1':
    print('ps: Permission denied', file=sys.stderr)
    sys.exit(1)
pid = sys.argv[sys.argv.index('-p') + 1]
if pid == '2147483647':
    sys.exit(1)  # A disappeared process: empty stdout/stderr.
print('/usr/bin/python3')
'''


class AQSSweepTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='dfine-aqs-sweep-test-')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.output = self.root / '实验 outputs'
        self.calls = self.root / 'simulated_calls.jsonl'
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.conda = self.bin / 'fake-conda'
        for path, body in ((self.conda, FAKE_CONDA), (self.bin / 'nvidia-smi', FAKE_GPU),
                           (self.bin / 'ps', FAKE_PS)):
            path.write_text('#!' + sys.executable + '\n' + body, encoding='utf-8')
            path.chmod(0o755)
        self.env = dict(os.environ, AQS_SWEEP_CONDA=str(self.conda),
                        AQS_SWEEP_CONDA_ENV='fake', AQS_SWEEP_GPU='0',
                        AQS_SWEEP_RUN_ID='1', AQS_SWEEP_OUTPUT_ROOT=str(self.output),
                        AQS_TEST_CALLS=str(self.calls))
        self.env['PATH'] = str(self.bin) + os.pathsep + self.env['PATH']
        # Do not inherit fault-injection settings from the caller.
        for key in ('AQS_TEST_FAIL_TAG', 'AQS_TEST_MISSING_TAG',
                    'AQS_TEST_PREFLIGHT_EXIT', 'AQS_TEST_GPU_PIDS', 'AQS_TEST_PS_DENIED'):
            self.env.pop(key, None)

    def run_queue(self, *args):
        return subprocess.run(['bash', str(SCRIPT), *args], env=self.env, cwd=ROOT,
                              capture_output=True, text=True, timeout=45)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_all_profiles_are_single_variable_l2_models_and_log_actual_temperature(self):
        reference = fresh_config('dfine_s_scbs_hrw_110_aqs_layer2.yml').yaml_cfg
        for group, (tag, tau, rho, temperature) in PROFILES.items():
            with self.subTest(group=group):
                filename = ('dfine_s_scbs_hrw_110_aqs_layer2.yml' if group == 'CONTROL'
                            else f'dfine_s_scbs_hrw_110_aqs_layer2_{tag}.yml')
                cfg = fresh_config(filename)
                expected, actual = copy.deepcopy(reference), copy.deepcopy(cfg.yaml_cfg)
                for item in (expected, actual):
                    item.pop('__include__', None)
                    item.pop('output_dir', None)
                expected['DFINETransformer'].update(aqs_threshold=tau,
                                                   aqs_residual_init=rho,
                                                   aqs_temperature=temperature)
                self.assertEqual(expected, actual)
                model = cfg.model
                decoder, refiner = model.decoder.decoder, model.decoder.decoder.query_cls_refiner
                self.assertEqual(decoder.aqs_layer_indices, (1,))
                self.assertEqual(len(decoder.aqs_refiners), 0)
                self.assertEqual(model.decoder.eval_idx, 2)
                self.assertAlmostEqual(refiner.threshold_logit.sigmoid().item(), tau, places=6)
                self.assertAlmostEqual(refiner.residual_scale.item(), rho, places=6)
                self.assertEqual(refiner.temperature, temperature)
                report = runtime_aqs_report(model, epoch=0)
                entry = report['sources']['model']['decoder.decoder.query_cls_refiner']
                self.assertEqual(entry['decoder_layer'], 2)
                self.assertEqual(entry['temperature'], temperature)

    def test_shell_syntax(self):
        self.assertEqual(subprocess.run(['bash', '-n', str(SCRIPT)]).returncode, 0)

    def test_dry_run_default_has_seven_commands_and_creates_nothing(self):
        result = self.run_queue('--dry-run')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('命令：'), 7)
        self.assertEqual(result.stdout.count('--seed=0'), 7)
        self.assertNotIn('SIMULATED TRAINING LOG', result.stdout)
        self.assertEqual(self.records(), [])
        self.assertFalse(self.output.exists())

    def test_default_sequential_all_seven_independent_outputs_and_live_logs(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = self.records()
        self.assertEqual(len(records), 7)
        self.assertEqual(len({item['output'] for item in records}), 7)
        self.assertEqual(result.stdout.count('SIMULATED TRAINING LOG'), 7)
        for item, (group, (tag, _, _, _)) in zip(records, list(PROFILES.items())[:7]):
            self.assertTrue(item['config'].endswith(f'layer2_{tag}.yml'), group)
            self.assertTrue(item['output'].endswith(f'layer2_{tag}_seed0_run1'), group)
            command = item['command']
            self.assertIn('--standalone', command)
            self.assertIn('--local-addr=127.0.0.1', command)
            self.assertIn('--seed=0', command)
            self.assertNotIn('-r', command)
            self.assertNotIn('-t', command)
            self.assertEqual(item['gpu'], '0')
            self.assertEqual(item['unbuffered'], '1')
            output = Path(item['output'])
            self.assertIn('SIMULATED TRAINING LOG', (output / 'console.log').read_text())
            self.assertIn(f'group={group}', (output / 'aqs_sweep_completed.txt').read_text())

    def test_selection_order_run_id_and_explicit_control(self):
        self.env['AQS_SWEEP_RUN_ID'] = '2'
        result = self.run_queue('T2', 'CONTROL', 'R1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([Path(x['config']).name for x in self.records()], [
            'dfine_s_scbs_hrw_110_aqs_layer2_temp020.yml',
            'dfine_s_scbs_hrw_110_aqs_layer2.yml',
            'dfine_s_scbs_hrw_110_aqs_layer2_r010.yml'])
        self.assertTrue(all(x['output'].endswith('seed0_run2') for x in self.records()))
        self.assertIn('layer2_control_seed0_run2', self.records()[1]['output'])

    def test_existing_later_output_aborts_before_any_training_and_preserves_file(self):
        existing = self.output / 'dfine_s_scbs_hrw_epochs110_aqs_layer2_r025_seed0_run1'
        existing.mkdir(parents=True)
        marker = existing / 'user_result.txt'
        marker.write_text('preserve this')
        result = self.run_queue()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.records(), [])
        self.assertEqual(marker.read_text(), 'preserve this')
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_failed_training_stops_before_next_group_and_preserves_partial_logs(self):
        self.env['AQS_TEST_FAIL_TAG'] = 'r020'
        result = self.run_queue('R1', 'R2', 'R3')
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertEqual(len(self.records()), 2)
        first, failed = [Path(x['output']) for x in self.records()]
        self.assertTrue((first / 'aqs_sweep_completed.txt').is_file())
        self.assertTrue((failed / 'console.log').is_file())
        self.assertFalse((failed / 'aqs_sweep_completed.txt').exists())
        self.assertIn('队列停止', result.stderr)
        self.assertFalse((self.output / 'dfine_s_scbs_hrw_epochs110_aqs_layer2_r025_seed0_run1').exists())

    def test_success_without_checkpoint_is_not_marked_complete(self):
        self.env['AQS_TEST_MISSING_TAG'] = 'r010'
        result = self.run_queue('R1', 'R2')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.records()), 1)
        self.assertFalse((Path(self.records()[0]['output']) / 'aqs_sweep_completed.txt').exists())

    def test_preflight_failure_creates_no_output(self):
        self.env['AQS_TEST_PREFLIGHT_EXIT'] = '19'
        result = self.run_queue('R1')
        self.assertEqual(result.returncode, 19)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.records(), [])

    def test_busy_gpu_stops_and_disappeared_gpu_pid_does_not(self):
        self.env['AQS_TEST_GPU_PIDS'] = str(os.getpid())
        busy = self.run_queue('R1')
        self.assertNotEqual(busy.returncode, 0)
        self.assertEqual(self.records(), [])
        self.assertEqual(list(self.output.iterdir()), [])
        self.env['AQS_TEST_GPU_PIDS'] = '2147483647'
        stale = self.run_queue('R1')
        self.assertEqual(stale.returncode, 0, stale.stdout + stale.stderr)
        self.assertEqual(len(self.records()), 1)

    def test_duplicate_unknown_group_and_invalid_run_id_do_not_launch(self):
        for args in (('R1', 'R1'), ('BAD',), ('--bad',)):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_queue(*args).returncode, 0)
                self.assertFalse(self.output.exists())
        self.env['AQS_SWEEP_RUN_ID'] = '0'
        self.assertNotEqual(self.run_queue('R1').returncode, 0)
        self.assertEqual(self.records(), [])

    def test_process_permission_error_is_not_mistaken_for_idle_gpu(self):
        self.env['AQS_TEST_GPU_PIDS'] = '12345'
        self.env['AQS_TEST_PS_DENIED'] = '1'
        result = self.run_queue('R1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('不能确认GPU空闲', result.stderr)
        self.assertEqual(self.records(), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
