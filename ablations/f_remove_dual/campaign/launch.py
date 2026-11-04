"""Guarded six-card launch. Does not import torch or allocate a CUDA context."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/root/miniconda3/envs/TS/bin/python'
VARIANTS = {'A': 0, 'B': 1, 'D': 2, 'E': 4, 'F': 5, 'G': 6}
FORBIDDEN_UUID = 'GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468'


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def gpu_state():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in raw.splitlines():
        index, uuid, used = [x.strip() for x in line.split(',')]
        result[int(index)] = dict(uuid=uuid, memory_MiB=int(used))
    return result


def assert_free(mapping):
    state = gpu_state()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    occupied = {line.split(',')[0].strip() for line in processes.splitlines()}
    for variant, index in VARIANTS.items():
        assert index != 3
        expected = mapping[str(index)]
        if expected == FORBIDDEN_UUID or state[index]['uuid'] != expected:
            raise RuntimeError('Physical GPU mapping changed')
        if expected in occupied or state[index]['memory_MiB'] > 256:
            raise RuntimeError(f'GPU {index} is occupied; refusing launch')


def env_for(uuid):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', PAPERLAB_PROGRESS='off',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='42', PYTHONUNBUFFERED='1')
    return env


def main():
    (ROOT / 'records').mkdir(exist_ok=True); (ROOT / 'logs').mkdir(exist_ok=True)
    lock = ROOT / 'records/launch.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    mapping = json.loads((ROOT / 'campaign/gpu_map.json').read_text())
    expected = json.loads((ROOT / 'bundle_manifest.json').read_text())
    for name, sha in expected['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Source fingerprint mismatch: ' + name)
    if not json.loads((ROOT / 'records/cpu_checks.json').read_text())['passed']:
        raise RuntimeError('CPU checks missing')
    audit = json.loads((ROOT / 'campaign/text_audit.json').read_text())
    if hashlib.sha256((ROOT / 'campaign/text_bank.pt').read_bytes()).hexdigest() != audit['bank_sha256']:
        raise RuntimeError('Text bank fingerprint mismatch')
    assert_free(mapping)
    preflights = []
    for variant, index in VARIANTS.items():
        log = open(ROOT / f'logs/preflight_{variant}.log', 'xb', buffering=0)
        proc = subprocess.Popen([PYTHON, '-u', '-m', 'campaign.preflight', '--variant', variant],
                                cwd=ROOT, env=env_for(mapping[str(index)]), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        log.close(); preflights.append((variant, proc))
    write(ROOT / 'records/launch_status.json', dict(stage='real_gpu_preflight',
          processes={v: p.pid for v, p in preflights}, started=datetime.datetime.now().astimezone().isoformat()))
    failures = []
    for variant, proc in preflights:
        if proc.wait() != 0: failures.append(variant)
    if failures:
        write(ROOT / 'records/launch_status.json', dict(stage='preflight_failed', variants=failures))
        raise RuntimeError('Preflight failed: ' + ','.join(failures))
    records = [json.loads((ROOT / f'records/preflight_{v}.json').read_text()) for v in VARIANTS]
    assert len({r['initial_state_sha256'] for r in records}) == 1
    assert len({json.dumps(r['manifest'], sort_keys=True) for r in records}) == 1
    # CUDA contexts have been destroyed by exiting preflight processes.
    time.sleep(2)
    assert_free(mapping)
    launched = {}
    for variant, index in VARIANTS.items():
        log = open(ROOT / f'logs/train_{variant}.log', 'xb', buffering=0)
        proc = subprocess.Popen([PYTHON, '-u', '-m', 'campaign.train', '--variant', variant],
                                cwd=ROOT, env=env_for(mapping[str(index)]), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        launched[variant] = dict(pid=proc.pid, physical_gpu=index, uuid=mapping[str(index)])
        print('LAUNCHED ' + json.dumps(dict(variant=variant, **launched[variant])), flush=True)
    write(ROOT / 'records/launch_status.json', dict(stage='training_launched', processes=launched,
          started=datetime.datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('LAUNCH_ERROR ' + repr(e), flush=True)
        raise
