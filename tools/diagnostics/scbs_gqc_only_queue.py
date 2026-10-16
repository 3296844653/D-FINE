"""Stage GQC-only now; wait/run in server tmux with an ISOLATED code snapshot.

No writes to the active D-FINE checkout, no weights/data copies, no job killing.
This is a cooperative local queue, not Slurm or an exclusive global GPU lease.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diagnostics.scbs_gqc_only import DEFAULT_ROOT, output_name

MANIFEST = 'tools/diagnostics/scbs_gqc_only_sync_files.txt'
SELF = 'tools/diagnostics/scbs_gqc_only_queue.py'
PROJECT = '/home/a5/MyProject/wq_project/D-FINE'
INTERVAL = 30


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def status(jobdir, state, **details):
    record = dict(state=state, updated_at=timestamp(), **details)
    path = jobdir/'queue_status.json'
    temporary = jobdir/f'.queue_status.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temporary.replace(path)
    print(f'[{record["updated_at"]}] {state} {json.dumps(details, ensure_ascii=False)}', flush=True)


def checked_relative(name):
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
        raise ValueError(f'非法清单路径：{name!r}')
    return relative


def fingerprints(root):
    names = (root/MANIFEST).read_text().splitlines()
    if len(names) != len(set(names)) or SELF not in names:
        raise ValueError('清单重复或缺少服务器队列程序')
    return {name: hashlib.sha256((root/checked_relative(name)).read_bytes()).hexdigest() for name in names}


def verify_payload(payload, job):
    for name, expected in job['sha256'].items():
        file = payload/checked_relative(name)
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f'待运行文件缺失或SHA256不符：{name}')


def resolve_conda(requested=None):
    # Non-interactive tmux/login shells may skip .bashrc's Conda setup.
    # Resolve on the SERVER; never forward the Mac's CONDA_EXE path.
    home = Path.home()
    candidates = [requested] if requested else [os.environ.get('CONDA_EXE'), shutil.which('conda'),
        '/home/a5/MySoftware/Anaconda3/bin/conda', str(home/'anaconda3/bin/conda'),
        str(home/'miniconda3/bin/conda'), '/opt/conda/bin/conda']
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    raise ValueError('服务器找不到Conda可执行文件；请用 --conda-exe 指定服务器上的完整路径，尚未创建任务')


def prepare_submission(job):
    """Remote stdlib-only reservation; never touch live model files."""
    project, jobdir = Path(job['project']), Path(job['job_dir'])
    if sys.version_info < (3, 9):
        raise ValueError('服务器python3需要3.9或更新版本')
    if not (project/'train.py').is_file() or not (project/'configs/dfine/dfine_s_scbs_hrw_110.yml').is_file():
        raise ValueError('服务器原始D-FINE项目/基线配置不存在')
    if shutil.which('tmux') is None or shutil.which('rsync') is None:
        raise ValueError('服务器需要tmux和rsync，尚未创建任务')
    if any(not (project/name).is_dir() for name in ('src', 'configs', 'tools/diagnostics')):
        raise ValueError('服务器项目源码/配置/诊断目录不完整')
    job['conda_exe'] = resolve_conda(job.get('conda_exe'))
    if Path(job['output']).exists() or Path(job['output']).is_symlink():
        raise ValueError('已有同名实验输出；请选择新run编号，不覆盖旧结果')
    sessions = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], capture_output=True, text=True)
    if job['session'] in sessions.stdout.splitlines():
        raise ValueError(f'已有同名tmux会话：{job["session"]}，请勿重复提交')
    if jobdir.parent.is_symlink():
        raise ValueError('待运行根目录是符号链接，停止提交')
    if jobdir.exists() or jobdir.is_symlink():
        raise ValueError(f'已有同名待运行任务：{jobdir}；请勿重复提交，新实验请选择未使用的run编号')
    jobdir.mkdir(parents=True, exist_ok=False)
    (jobdir/'payload').mkdir()
    (jobdir/'submission.json').write_text(json.dumps(job, ensure_ascii=False, indent=2))
    status(jobdir, 'uploading', session=job['session'])


def dispatch(jobdir):
    """Remote dispatch after payload verification; no training on SSH itself."""
    job = json.loads((jobdir/'submission.json').read_text())
    verify_payload(jobdir/'payload', job)
    controller = jobdir/'payload'/SELF
    worker = shlex.join(['python3', '-u', str(controller), 'worker', '--job-dir', str(jobdir)])
    pipeline = 'set -o pipefail; '+worker+' 2>&1 | tee -a '+shlex.quote(str(jobdir/'queue.log'))
    command = shlex.join(['bash', '-lc', pipeline])
    status(jobdir, 'submitted', session=job['session'])
    subprocess.run(['tmux', 'new-session', '-d', '-s', job['session'], command], check=True)
    time.sleep(.5)
    alive = subprocess.run(['tmux', 'has-session', '-t', job['session']], capture_output=True).returncode == 0
    record = json.loads((jobdir/'queue_status.json').read_text())
    if not alive and record['state'] != 'complete':
        raise RuntimeError(f'服务器队列未保持运行：{record}；查看 {jobdir}/queue.log')
    print(f'已进入服务器队列：{job["session"]}\n状态：{jobdir}/queue_status.json\n日志：{jobdir}/queue.log', flush=True)


def busy_project(project, proc_root=Path('/proc')):
    """Keep waiting while an existing project queue exists, even in GPU gaps."""
    project = project.resolve()
    active = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            words = [word.decode(errors='replace') for word in (proc/'cmdline').read_bytes().split(b'\0') if word]
            cwd = (proc/'cwd').resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        # Sibling GQC queue workers are serialized by the project queue lock.
        # They must not wait for each other as if they were existing training.
        if any(Path(word).name == Path(SELF).name for word in words):
            continue
        scoped = cwd == project or project in cwd.parents or any(str(project) in word for word in words)
        training = any(Path(word).name == 'train.py' or
            word.startswith('tools.diagnostics.') or
            ('tools/diagnostics/' in word and word.endswith(('.py', '.sh'))) for word in words)
        if scoped and training:
            active.append(int(proc.name))
    return sorted(active)


@contextmanager
def queue_lock(jobdir, sleep=time.sleep):
    import fcntl
    # All GQC submissions for this project share one server-local lock.
    with (jobdir.parent/'.gqc_only_queue.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status(jobdir, 'waiting_queue_lock')
                sleep(INTERVAL)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def snapshot_workspace(jobdir, job):
    """Copy only small code/config directories, NEVER the active checkout itself."""
    project, payload, workspace = Path(job['project']), jobdir/'payload', jobdir/'workspace'
    verify_payload(payload, job)
    workspace.mkdir(exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store')
    for relative in ('src', 'configs', 'tools/diagnostics'):
        shutil.copytree(project/relative, workspace/relative, symlinks=True, ignore=ignore)
    shutil.copy2(project/'train.py', workspace/'train.py')
    # Dataset access stays at the original files. No copying large data/weights.
    for relative in ('dataset', 'datasets', 'data', 'weight', 'weights'):
        source = project/relative
        if source.exists():
            (workspace/relative).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    for name in job['sha256']:
        relative = checked_relative(name); target = workspace/relative
        if target.is_symlink() or not target.parent.resolve().is_relative_to(workspace.resolve()):
            raise ValueError(f'副本中存在越界符号链接，不覆盖：{target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(payload/relative, target)
    verify_payload(workspace, job)
    return workspace


def train_command(workspace, job):
    args = job['arguments']
    command = ['bash', str(workspace/'tools/diagnostics/run_scbs_gqc_only_110.sh'), args['run'],
               '--seed', str(args['seed']), '--gpu', str(args['gpu']), '--port', str(args['port']),
               '--root', args['root'], '--wait-gpu']
    if args['diagnostics']:
        command.append('--diagnostics')
    return command


def wait_until_ready(jobdir, job, query_gpu, project_jobs=None, sleep=None):
    project_jobs, sleep = project_jobs or busy_project, sleep or time.sleep
    while True:
        active = project_jobs(Path(job['project']))
        if active:
            status(jobdir, 'waiting_project', PIDs=active)
        else:
            processes = query_gpu(job['arguments']['gpu'])
            if not processes:
                return
            status(jobdir, 'waiting_gpu', processes=processes)
        sleep(INTERVAL)


def worker(jobdir):
    jobdir = jobdir.resolve()
    job = json.loads((jobdir/'submission.json').read_text())
    if Path(job['job_dir']).resolve() != jobdir:
        raise ValueError('任务目录与提交记录不一致')
    try:
        verify_payload(jobdir/'payload', job)
        with queue_lock(jobdir):
            status(jobdir, 'preparing_isolated_workspace')
            workspace = snapshot_workspace(jobdir, job)
            # Import stdlib-only GPU checker from the staged payload, not live src.
            spec = importlib.util.spec_from_file_location('gqc_queue_gpu', jobdir/'payload/tools/diagnostics/wait_for_gpu.py')
            gpu = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpu)
            wait_until_ready(jobdir, job, gpu.compute_processes)
            if Path(job['output']).exists() or Path(job['output']).is_symlink():
                raise ValueError('排队期间出现同名实验输出；停止，不覆盖、不自动续训')
            cmd = train_command(workspace, job)
            status(jobdir, 'running', workspace=str(workspace), output=job['output'], command=cmd)
            env = dict(os.environ, SCBS_GQC_ENV=job['environment'], SCBS_GQC_CONDA=job['conda_exe'], PYTHONUNBUFFERED='1')
            subprocess.run(cmd, cwd=workspace, env=env, check=True)
            status(jobdir, 'complete', output=job['output'])
    except BaseException as error:
        status(jobdir, 'failed', error=f'{type(error).__name__}: {error}', output=job['output'])
        raise


def remote_python(target, options, code, data):
    encoded = base64.urlsafe_b64encode(json.dumps(data, ensure_ascii=False).encode()).decode()
    prefix = 'import base64,json,sys\nrequest=json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode())\n'
    subprocess.run(['ssh', *options, target, 'python3', '-', encoded], input=prefix+code, text=True, check=True)


def submit(args):
    if ROOT != Path('/Users/wq/Desktop/D-FINE'):
        raise ValueError('请从Mac桌面D-FINE提交')
    if not re.fullmatch(r'run[1-9][0-9]*', args.run) or args.seed < 0 or args.gpu < 0 or not 1024 <= args.port <= 65535:
        raise ValueError('需要run1/run2、非负seed/GPU及1024..65535端口')
    if (not Path(args.root).is_absolute() or args.ssh_target.startswith('-')
            or not re.fullmatch(r'[A-Za-z0-9_@.:\[\]-]+', args.ssh_target)):
        raise ValueError('输出根目录须为绝对路径；SSH目标格式不符')
    if not re.fullmatch(r'[\w.-]+', args.environment):
        raise ValueError('Conda环境名格式不符')
    jobdir = Path(PROJECT).parent/'D-FINE-pending'/f'GQC_only_seed{args.seed}_{args.run}'
    job = dict(schema=1, submitted_at=timestamp(), project=PROJECT, job_dir=str(jobdir),
        output=str(Path(args.root)/output_name(args.seed, args.run)),
        session=f'scbs_gqc_only_seed{args.seed}_{args.run}', environment=args.environment,
        conda_exe=getattr(args, 'conda_exe', None),
        arguments=dict(run=args.run, seed=args.seed, gpu=args.gpu, port=args.port, root=args.root, diagnostics=args.diagnostics),
        sha256=fingerprints(ROOT))
    print(f'待运行目录：{jobdir}\n输出：{job["output"]}\n上传后由服务器tmux等待旧D-FINE队列和GPU；不覆盖当前项目源码。', flush=True)
    if args.dry_run:
        print('仅本地演练：不连接服务器、不创建目录、不提交任务。', flush=True)
        return
    # One password-capable connection reused for reserve/upload/dispatch. The
    # short /tmp socket avoids macOS UNIX-domain socket path length limits.
    with tempfile.TemporaryDirectory(prefix='gqc-ssh-', dir='/tmp') as temporary:
        options = ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=60', '-o', 'ConnectTimeout=15',
                   '-o', f'ControlPath={temporary}/socket']
        try:
            # Serialize just the stdlib helper functions needed BEFORE upload.
            import inspect
            bootstrap = 'import json,shutil,subprocess,os\nfrom pathlib import Path\nfrom datetime import datetime,timezone\n'
            bootstrap += '\n\n'.join(inspect.getsource(fn) for fn in (timestamp, status, resolve_conda, prepare_submission))
            remote_python(args.ssh_target, options, bootstrap+'\nprepare_submission(request)\n', job)
            subprocess.run(['rsync', '-av', '--checksum', '--relative', '--files-from='+str(ROOT/MANIFEST),
                '-e', shlex.join(['ssh', *options]), str(ROOT)+'/', f'{args.ssh_target}:{jobdir}/payload/'], check=True)
            code = ('import importlib.util\nfrom pathlib import Path\n'
                f'file=Path(request["job_dir"])/"payload"/{SELF!r}\n'
                'spec=importlib.util.spec_from_file_location("gqc_queue_dispatch",file)\n'
                'module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n'
                'module.dispatch(Path(request["job_dir"]))\n')
            remote_python(args.ssh_target, options, code, job)
        finally:
            try:
                subprocess.run(['ssh', *options, '-O', 'exit', args.ssh_target], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass  # Do not mask an upload/dispatch error with socket cleanup.
    print('服务器排队提交成功。现在可以关闭Mac；不需要再手动同步或启动训练。', flush=True)
    print(f'查看进度（服务器）：tmux attach -t {job["session"]}\n或：tail -f {jobdir}/queue.log', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    submit_parser = modes.add_parser('submit')
    submit_parser.add_argument('run', nargs='?', default='run1')
    submit_parser.add_argument('--seed', type=int, default=0)
    submit_parser.add_argument('--gpu', type=int, default=int(os.environ.get('SCBS_GQC_GPU', '0')))
    submit_parser.add_argument('--port', type=int, default=int(os.environ.get('SCBS_GQC_PORT', '7785')))
    submit_parser.add_argument('--root', default=os.environ.get('SCBS_GQC_ROOT', DEFAULT_ROOT))
    submit_parser.add_argument('--environment', default=os.environ.get('SCBS_GQC_ENV', 'wq'))
    submit_parser.add_argument('--conda-exe', help='服务器上的Conda完整路径；默认在服务器自动查找')
    submit_parser.add_argument('--ssh-target', default=os.environ.get('SCBS_GQC_SSH_TARGET', 'a5@100.76.151.67'))
    submit_parser.add_argument('--diagnostics', action='store_true')
    submit_parser.add_argument('--dry-run', action='store_true')
    worker_parser = modes.add_parser('worker')
    worker_parser.add_argument('--job-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'submit':
        submit(args)
    else:
        worker(args.job_dir)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f'错误：{error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
