"""Wait for compute processes to leave one GPU. Standard library only."""
import argparse
import csv
import io
import subprocess
import sys
import time
from pathlib import Path

DESKTOP_PROCESSES = {'Xorg', 'gnome-remote-desktop-daemon', 'gnome-shell'}


class GPUQueryError(RuntimeError):
    pass


def compute_processes(gpu):
    try:
        result = subprocess.run(
            ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name',
             '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GPUQueryError(f'无法检查GPU：{error}') from error
    if result.returncode:
        raise GPUQueryError(f'nvidia-smi检查失败：{(result.stderr or result.stdout).strip()}')
    processes = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 2:
            raise GPUQueryError(f'无法解析GPU进程记录：{row!r}')
        pid, name = (field.strip() for field in row)
        if not pid.isdigit() or int(pid) <= 0:
            raise GPUQueryError(f'无法确认GPU进程编号：{pid!r}')
        # A missing process name still represents occupied GPU memory.
        if Path(name).name not in DESKTOP_PROCESSES:
            processes.append((int(pid), name or '[unknown]'))
    return processes


def wait_for_gpu(gpu, wait=False, interval=30, idle_checks=2,
                 query=None, sleep=None, emit=None):
    if type(gpu) is not int or gpu < 0:
        raise ValueError('GPU编号必须是非负整数')
    if type(interval) is not int or interval < 1:
        raise ValueError('检查间隔必须是正整数秒')
    if type(idle_checks) is not int or idle_checks < 1:
        raise ValueError('连续空闲检查次数必须是正整数')
    query = query or compute_processes
    sleep = sleep or time.sleep
    emit = emit or (lambda message: print(message, flush=True))
    consecutive = 0
    while True:
        busy = query(gpu)
        if busy:
            consecutive = 0
            description = '; '.join(f'{pid}: {name}' for pid, name in busy)
            if not wait:
                raise RuntimeError(f'GPU{gpu}仍有计算进程：{description}')
            emit(f'等待GPU{gpu}释放；{interval}秒后复查。占用：{description}')
        else:
            consecutive += 1
            if not wait or consecutive >= idle_checks:
                emit(f'GPU{gpu}计算进程检查通过，开始后续诊断。')
                return
            emit(f'GPU{gpu}空闲确认 {consecutive}/{idle_checks}；{interval}秒后复查。')
        sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--idle-checks', type=int, default=2)
    args = parser.parse_args()
    try:
        wait_for_gpu(args.gpu, args.wait, args.interval, args.idle_checks)
    except KeyboardInterrupt:
        print('已取消GPU等待。', file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
