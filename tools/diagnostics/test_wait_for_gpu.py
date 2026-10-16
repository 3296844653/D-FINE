"""GPU queue checks use simulated process lists, never a real GPU or sleep."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.diagnostics import wait_for_gpu as queue


class GPUWaitTests(unittest.TestCase):
    def test_parser_ignores_only_known_desktop_names(self):
        output = ('1340, /usr/lib/xorg/Xorg\n'
                  '1585, /usr/libexec/gnome-remote-desktop-daemon\n'
                  '2000, /tmp/gnome-shell-training\n'
                  '3000, [No data]\n'
                  '4000, /env/python\n')
        with patch.object(queue.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=output,stderr='')) as run:
            self.assertEqual(queue.compute_processes(0),[(2000,'/tmp/gnome-shell-training'),(3000,'[No data]'),(4000,'/env/python')])
            self.assertIn('0',run.call_args.args[0])

    def test_busy_then_two_idle_checks_before_return(self):
        snapshots = iter([[(5,'train.py')],[],[]])
        sleeps,messages = [],[]
        queue.wait_for_gpu(0,wait=True,query=lambda gpu:next(snapshots),sleep=sleeps.append,emit=messages.append)
        self.assertEqual(sleeps,[30,30])
        self.assertIn('train.py',messages[0])
        self.assertIn('1/2',messages[1])
        self.assertIn('开始后续诊断',messages[-1])

    def test_busy_return_resets_idle_confirmation(self):
        snapshots = iter([[],[(7,'llama-server')],[],[]])
        sleeps = []
        queue.wait_for_gpu(0,wait=True,interval=5,query=lambda gpu:next(snapshots),sleep=sleeps.append,emit=lambda line:None)
        self.assertEqual(sleeps,[5,5,5])

    def test_nonwaiting_busy_fails_and_free_returns_immediately(self):
        sleeps = []
        with self.assertRaisesRegex(RuntimeError,'计算进程'):
            queue.wait_for_gpu(0,query=lambda gpu:[(5,'train.py')],sleep=sleeps.append)
        queue.wait_for_gpu(0,query=lambda gpu:[],sleep=sleeps.append,emit=lambda line:None)
        self.assertEqual(sleeps,[])

    def test_query_errors_and_bad_records_fail_closed(self):
        for result in (SimpleNamespace(returncode=1,stdout='',stderr='driver unavailable'),
                       SimpleNamespace(returncode=0,stdout='unexpected output\n',stderr=''),
                       SimpleNamespace(returncode=0,stdout='[N/A], python\n',stderr='')):
            with patch.object(queue.subprocess,'run',return_value=result):
                with self.assertRaises(queue.GPUQueryError): queue.compute_processes(0)
        with patch.object(queue.subprocess,'run',side_effect=subprocess.TimeoutExpired('nvidia-smi',15)):
            with self.assertRaises(queue.GPUQueryError): queue.compute_processes(0)
        sleeps = []
        def bad_query(gpu): raise queue.GPUQueryError('query failed')
        with self.assertRaises(queue.GPUQueryError):
            queue.wait_for_gpu(0,wait=True,query=bad_query,sleep=sleeps.append)
        self.assertEqual(sleeps,[])

    def test_invalid_settings_and_cancellation_do_not_start_work(self):
        for values in (dict(gpu=-1),dict(gpu=0,interval=0),dict(gpu=0,idle_checks=0)):
            with self.assertRaises(ValueError): queue.wait_for_gpu(**values)
        def cancel(seconds): raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            queue.wait_for_gpu(0,wait=True,query=lambda gpu:[(4,'train.py')],sleep=cancel,emit=lambda line:None)

    def test_launcher_passes_wait_option_and_does_not_launch_after_gpu_failure(self):
        script = ROOT/'tools/diagnostics/run_scbs_AQS_WholeModelAudit.sh'
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fake = folder/'conda-fixture'
            calls = folder/'calls.txt'
            fake.write_text('#!/usr/bin/env python3\n'
                'import os,sys\nfrom pathlib import Path\n'
                'with Path(os.environ["QUEUE_FIXTURE_CALLS"]).open("a") as stream: stream.write(" ".join(sys.argv[1:])+"\\n")\n'
                'if "tools/diagnostics/wait_for_gpu.py" in sys.argv: sys.exit(int(os.environ["QUEUE_FIXTURE_GPU_CODE"]))\n'
                'print("synthetic audit command accepted")\n')
            fake.chmod(0o700)
            env = dict(os.environ,AQS_AUDIT_OUTPUT_ROOT=str(folder),AQS_AUDIT_CONDA=str(fake),
                       QUEUE_FIXTURE_CALLS=str(calls),QUEUE_FIXTURE_GPU_CODE='9')
            failed = subprocess.run(['bash',str(script),'--wait-gpu','audit1'],env=env,capture_output=True,text=True)
            self.assertEqual(failed.returncode,9,failed.stderr)
            self.assertIn('--wait',calls.read_text())
            self.assertNotIn('scbs_whole_model_audit.py',calls.read_text())
            self.assertFalse((folder/'AQS_WholeModelAudit_audit1').exists())
            self.assertFalse((folder/'AQS_WholeModelAudit_audit1_console.log').exists())
            env['QUEUE_FIXTURE_GPU_CODE']='0'
            successful = subprocess.run(['bash',str(script),'--wait-gpu','audit1'],env=env,capture_output=True,text=True)
            self.assertEqual(successful.returncode,0,successful.stderr)
            self.assertIn('scbs_whole_model_audit.py',calls.read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
