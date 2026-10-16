"""Stdlib-only queue tests: NO SSH, live tmux, GPU use or real training."""
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.diagnostics.scbs_gqc_only_queue as queue


def fixture(base, run='run1'):
    project = base/'D-FINE'
    for relative in ('src', 'configs/dfine', 'tools/diagnostics', 'dataset', 'weight', '.git', 'output'):
        (project/relative).mkdir(parents=True, exist_ok=True)
    original = {'src/base.py': 'original active code\n', 'train.py': '# original entry\n',
        'configs/dfine/dfine_s_scbs_hrw_110.yml': 'epochs: 110\n',
        'tools/diagnostics/existing.py': '# existing\n', 'dataset/image.bin': 'original data',
        'weight/pretrained.pth': 'original pretrained weights', '.git/secret': 'must not copy',
        'output/old.txt': 'existing results'}
    for name, text in original.items():
        (project/name).write_text(text)
    conda = base/'conda-fixture'; conda.write_text('#!/bin/sh\nexit 0\n'); conda.chmod(0o755)
    jobdir = base/'D-FINE-pending'/f'GQC_only_seed0_{run}'; (jobdir/'payload').mkdir(parents=True)
    files = {'src/base.py': '# NEW isolated code\n', queue.SELF: '# staged controller\n',
        'tools/diagnostics/run_scbs_gqc_only_110.sh': '#!/bin/bash\nexit 0\n',
        'tools/diagnostics/wait_for_gpu.py': 'def compute_processes(gpu):\n    return []\n'}
    for name, text in files.items():
        file = jobdir/'payload'/name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(text)
    job = dict(schema=1, project=str(project), job_dir=str(jobdir), session=f'scbs_gqc_only_seed0_{run}',
        environment='wq', conda_exe=str(conda), output=str(base/'results'/queue.output_name(0, run)),
        arguments=dict(run=run, seed=0, gpu=0, port=7785, root=str(base/'results'), diagnostics=False),
        sha256={name: hashlib.sha256((jobdir/'payload'/name).read_bytes()).hexdigest() for name in files})
    (jobdir/'submission.json').write_text(json.dumps(job))
    return project, jobdir, job, original


class GQCQueueTests(unittest.TestCase):
    def test_local_dry_run_no_network_no_outputs(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-dry-') as folder:
            env = dict(os.environ, SCBS_GQC_ROOT=folder, SCBS_GQC_CONDA='/no/conda')
            result = subprocess.run(['bash', str(ROOT/'tools/diagnostics/queue_scbs_gqc_only_110.sh'),
                'run1', '--dry-run'], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('不连接服务器', result.stdout)
            self.assertIn('D-FINE-pending/GQC_only_seed0_run1', result.stdout)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_snapshot_overlays_isolated_files_and_preserves_active_project(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-copy-') as folder:
            project, jobdir, job, original = fixture(Path(folder))
            workspace = queue.snapshot_workspace(jobdir, job)
            self.assertEqual((workspace/'src/base.py').read_text(), '# NEW isolated code\n')
            self.assertFalse((workspace/'.git').exists()); self.assertFalse((workspace/'output').exists())
            self.assertTrue((workspace/'dataset').is_symlink()); self.assertTrue((workspace/'weight').is_symlink())
            for name, text in original.items(): self.assertEqual((project/name).read_text(), text)
            with self.assertRaises(FileExistsError): queue.snapshot_workspace(jobdir, job)
            for bad in ('/absolute', '../escape', 'src/../../escape', ''):
                with self.assertRaises(ValueError): queue.checked_relative(bad)
            (jobdir/'payload/src/base.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'SHA256'): queue.verify_payload(jobdir/'payload', job)

    def test_parent_symlink_is_rejected_before_writing_outside_workspace(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-link-') as folder:
            base = Path(folder); project, jobdir, job, original = fixture(base)
            outside = base/'outside'; outside.mkdir()
            (project/'src/external').symlink_to(outside, target_is_directory=True)
            name = 'src/external/must-not-create/file.py'
            source = jobdir/'payload'/name; source.parent.mkdir(parents=True); source.write_text('new code')
            job['sha256'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, '越界符号链接'): queue.snapshot_workspace(jobdir, job)
            self.assertEqual(list(outside.iterdir()), [])

    def test_waits_for_existing_queue_before_checking_gpu(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-wait-') as folder:
            _, jobdir, job, _ = fixture(Path(folder))
            projects = unittest.mock.Mock(side_effect=[[11, 12], [11], [], []])
            gpu = unittest.mock.Mock(side_effect=[[(99, 'llama')], []]); sleeps = []
            with redirect_stdout(io.StringIO()):
                queue.wait_until_ready(jobdir, job, gpu, project_jobs=projects, sleep=sleeps.append)
            self.assertEqual(projects.call_count, 4); self.assertEqual(gpu.call_count, 2)
            self.assertEqual(sleeps, [30, 30, 30])

    def test_proc_scan_counts_idle_queued_controller_not_other_projects_or_siblings(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-proc-') as folder:
            base = Path(folder); project, _, _, _ = fixture(base)
            other = base/'other'; other.mkdir(); proc = base/'proc'; proc.mkdir()
            cases = [(100001, project, ['python', 'tools/diagnostics/scbs_boxlocal_five.py', 'all']),
                     (100002, project, ['python', 'train.py']),
                     (100003, other, ['python', 'train.py']),
                     (100004, project, ['python', '/pending/'+queue.SELF, 'worker']),
                     (100005, project, ['python', '-m', 'tools.diagnostics.scbs_behavior_four_queue'])]
            for pid, cwd, words in cases:
                path = proc/str(pid); path.mkdir(); (path/'cwd').symlink_to(cwd, target_is_directory=True)
                (path/'cmdline').write_bytes(b'\0'.join(word.encode() for word in words))
            self.assertEqual(queue.busy_project(project, proc), [100001, 100002, 100005])

    def test_worker_success_failure_and_output_collision_never_touch_shared_code(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-worker-') as folder:
            project, jobdir, job, original = fixture(Path(folder))
            with patch.object(queue, 'busy_project', return_value=[]), \
                 patch.object(queue.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as train, \
                 redirect_stdout(io.StringIO()):
                queue.worker(jobdir)
            self.assertEqual(train.call_count, 1)
            command = train.call_args.args[0]
            self.assertIn(str(jobdir.resolve()/'workspace/tools/diagnostics/run_scbs_gqc_only_110.sh'), command)
            self.assertIn('--wait-gpu', command)
            self.assertEqual(json.loads((jobdir/'queue_status.json').read_text())['state'], 'complete')
            for name, text in original.items(): self.assertEqual((project/name).read_text(), text)
        for kind in ('failure', 'collision'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='gqc-queue-fail-') as folder:
                project, jobdir, job, original = fixture(Path(folder))
                if kind == 'collision':
                    Path(job['output']).mkdir(parents=True); (Path(job['output'])/'old').write_text('keep')
                with patch.object(queue, 'busy_project', return_value=[]), \
                     patch.object(queue.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['fixture'])) as train, \
                     redirect_stdout(io.StringIO()), self.assertRaises((ValueError, subprocess.CalledProcessError)):
                    queue.worker(jobdir)
                self.assertEqual(train.call_count, 0 if kind == 'collision' else 1)
                self.assertEqual(json.loads((jobdir/'queue_status.json').read_text())['state'], 'failed')
                for name, text in original.items(): self.assertEqual((project/name).read_text(), text)

    def test_remote_reservation_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-prepare-') as folder:
            project, existing, job, _ = fixture(Path(folder))
            new = existing.parent/'new-job'; job['job_dir'] = str(new)
            completed = subprocess.CompletedProcess([], 1, stdout='', stderr='no sessions')
            with patch.object(queue.shutil, 'which', return_value='/usr/bin/fixture'), \
                 patch.object(queue.subprocess, 'run', return_value=completed), redirect_stdout(io.StringIO()):
                queue.prepare_submission(job)
                self.assertTrue((new/'submission.json').is_file())
                with self.assertRaisesRegex(ValueError, '同名待运行任务'): queue.prepare_submission(job)
                duplicate = dict(job, job_dir=str(existing.parent/'another'))
                Path(job['output']).mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, '同名实验输出'): queue.prepare_submission(duplicate)

    def test_dispatch_verifies_upload_then_uses_detached_tmux(self):
        with tempfile.TemporaryDirectory(prefix='gqc-queue-dispatch-') as folder:
            _, jobdir, job, _ = fixture(Path(folder))
            with patch.object(queue.time, 'sleep'), patch.object(queue.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 0)) as remote, redirect_stdout(io.StringIO()):
                queue.dispatch(jobdir)
            self.assertEqual(remote.call_args_list[0].args[0][:4], ['tmux', 'new-session', '-d', '-s'])
            self.assertIn(job['session'], remote.call_args_list[0].args[0])
            shell_command = remote.call_args_list[0].args[0][-1]
            self.assertIn('tee -a', shell_command); self.assertIn('worker', shell_command)
            (jobdir/'payload/src/base.py').write_text('bad upload')
            with patch.object(queue.subprocess, 'run') as remote, self.assertRaises(ValueError): queue.dispatch(jobdir)
            remote.assert_not_called()

    def test_project_lock_prevents_two_gqc_jobs_running_at_once(self):
        import fcntl
        with tempfile.TemporaryDirectory(prefix='gqc-queue-lock-') as folder:
            _, jobdir, _, _ = fixture(Path(folder))
            with queue.queue_lock(jobdir):
                with (jobdir.parent/'.gqc_only_queue.lock').open('a') as sibling:
                    with self.assertRaises(BlockingIOError): fcntl.flock(sibling, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_remote_python_encodes_unicode_and_spaces_without_shell_interpolation(self):
        request = {'root': '/media/a5/测试 结果', 'seed': 0}
        with patch.object(queue.subprocess, 'run') as remote:
            queue.remote_python('a5@host', [], 'print(request)\n', request)
        command = remote.call_args.args[0]
        self.assertEqual(json.loads(base64.urlsafe_b64decode(command[-1])), request)
        self.assertIn('print(request)', remote.call_args.kwargs['input'])

    def test_submit_only_uploads_pending_not_live_checkout(self):
        args = SimpleNamespace(run='run1', seed=0, gpu=0, port=7785, root='/media/results',
            ssh_target='a5@100.76.151.67', environment='wq', diagnostics=False, dry_run=False)
        with patch.object(queue, 'remote_python') as rpc, patch.object(queue.subprocess, 'run',
                return_value=subprocess.CompletedProcess([], 0)) as remote, redirect_stdout(io.StringIO()):
            queue.submit(args)
        self.assertEqual(rpc.call_count, 2)
        upload = remote.call_args_list[0].args[0]
        self.assertEqual(upload[0], 'rsync')
        self.assertTrue(upload[-1].endswith('/D-FINE-pending/GQC_only_seed0_run1/payload/'))
        self.assertNotEqual(upload[-1], 'a5@100.76.151.67:'+queue.PROJECT+'/')

    def test_actual_bootstrap_and_dispatch_code_with_fake_remote_transport(self):
        """Execute transmitted Python locally; mock SSH/tmux/rsync, not the code."""
        with tempfile.TemporaryDirectory(prefix='gqc-queue-rpc-') as folder:
            project, _, old_job, original = fixture(Path(folder))
            args = SimpleNamespace(run='run2', seed=0, gpu=0, port=7785, root=str(Path(folder)/'results'),
                ssh_target='a5@fixture', environment='wq', conda_exe=old_job['conda_exe'], diagnostics=False, dry_run=False)
            def execute(command, **kwargs):
                if command[0] == 'rsync':
                    destination = Path(command[-1].split(':', 1)[1])
                    for name in (ROOT/queue.MANIFEST).read_text().splitlines():
                        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(ROOT/name, target)
                elif command[0] not in ('tmux', 'ssh'):
                    raise AssertionError(f'Unexpected external command: {command}')
                return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
            def rpc(target, options, code, request):
                exec(compile(code, 'transmitted-python', 'exec'), {'request': request, 'sys': sys})
            with patch.object(queue, 'PROJECT', str(project)), patch.object(queue, 'remote_python', side_effect=rpc), \
                 patch.object(queue.shutil, 'which', return_value='/usr/bin/fixture'), \
                 patch.object(queue.time, 'sleep'), patch.object(queue.subprocess, 'run', side_effect=execute), \
                 redirect_stdout(io.StringIO()):
                queue.submit(args)
            jobdir = project.parent/'D-FINE-pending/GQC_only_seed0_run2'
            saved = json.loads((jobdir/'submission.json').read_text())
            self.assertEqual(saved['conda_exe'], str(Path(old_job['conda_exe']).resolve()))
            self.assertEqual(json.loads((jobdir/'queue_status.json').read_text())['state'], 'submitted')
            queue.verify_payload(jobdir/'payload', saved)
            for name, text in original.items(): self.assertEqual((project/name).read_text(), text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
