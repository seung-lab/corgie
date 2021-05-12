import glob
import os
import yaml
import json
import sys
import glob
import shutil
import atexit
import click

import numpy as np
from multiprocessing import Process
from functools import partial

from click.testing import CliRunner
from corgie.main import cli
from corgie.worker import worker_f

test_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../tests'
)

runner = CliRunner()

def run_test_command(test_name, original_stack_path, distributed=False):
    spec_path = os.path.join(test_path, f'test_specs/{test_name}.yaml')
    dst_path = os.path.join(test_path, f'test_data/tmp/{test_name}')
    gt_path = os.path.join(test_path, f'test_data/gt/{test_name}')

    shutil.rmtree(dst_path, ignore_errors=True)
    result = run_command(spec_path, dst_path, distributed=distributed)
    assert result.exit_code == 0
    assert_folder_equality(gt_path, dst_path)


def close_processes(ps):
    for p in ps:
        p.terminate()
    for p in ps:
        p.join()

def run_command(spec_path, dst_path, distributed=False):
    with open(spec_path, 'r') as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    original_stack_path = os.path.join(
        test_path,
        f'test_data/original/{spec["original_stack_name"]}'
    )

    if distributed:
        q_path = 'fq://' + os.path.join(test_path, 'test_data/queue/')
        shutil.rmtree(q_path, ignore_errors=True)
        worker_args = ['-l', 200, '-q', q_path, '-v']
        '''worker_process = Process(
            target=partial(
                worker,
                lease_seconds=240,
                queue_name=q_path,
                verbose=True
            )
        )
        worker_process.start()
        worker_process = Process(
            target=partial(
                runner.invoke,
                worker,
                worker_args
            )
        )
        worker_process = Process(
            target=partial(
                worker_f,
                lease_seconds=240,
                queue_name=q_path,
                verbose=True,
                completion_queue_name=None,
                sqs_queue_region="us-east-1",
                restart_from_checkpoint_file=None
            )
        )

        worker_process.start()
        atexit.register(
            partial(
                close_processes,
                [worker_process]
            )
        )'''


    args = [spec['command']]
    for k, v in spec['params'].items():
        if isinstance(v, list) and not isinstance(v, str):
            for vv in v:
                args.append(f'--{k}')
                if isinstance(vv, str):
                    vv = vv.replace('{ORIGINAL_STACK_PATH}', original_stack_path)
                    vv = vv.replace('{DST_FOLDER}', dst_path)
                args.append(vv)
        else:
            args.append(f'--{k}')
            if isinstance(v, str):
                v = v.replace('{ORIGINAL_STACK_PATH}', original_stack_path)
                v = v.replace('{DST_FOLDER}', dst_path)
            args.append(v)
    if distributed:
        args.append('-q')
        args.append(q_path)
    print (args)
    result = runner.invoke(cli, args, catch_exceptions=False)
    return result


def get_children(p):
    return [os.path.basename(x) for x in glob.glob(os.path.join(p, '*'))]

def assert_folder_equality(path1, path2, parent=None):
    subs1 = get_children(path1)
    subs2 = get_children(path2)

    for s in subs1:
        assert s in subs2
    for s in subs2:
        assert s in subs1
    # if it's with info, assume that it's data

    if 'info' in subs1:
        if parent not in ['img', 'layer', 'mask']:
            # WARNING: only properly subfolderd cv's will be checked
            return
        else:
            mode = parent

        match_files(
            os.path.join(path1, 'info'),
            os.path.join(path2, 'info'),
            'json'
        )
        with open(os.path.join(path1, 'info'), 'r') as f:
            info = json.load(f)
        encoding = info['scales'][0]['encoding']
        subs1.remove('info')
        subs2.remove('info')
        for s in subs1:
            s1_abs = os.path.join(path1, s)
            s2_abs = os.path.join(path2, s)

            if os.path.isdir(s1_abs):
                fs1 = get_children(s1_abs)
                fs2 = get_children(s2_abs)

                for f in fs1:
                    assert f in fs2

                for f in fs2:
                    assert f in fs1

                for f in fs1:
                    # WARNING: we don't compare cv's in other's cv's folder
                    if f.endswith('.gz'):
                        f1_abs = os.path.join(s1_abs, f)
                        f2_abs = os.path.join(s2_abs, f)
                        for sc in info['scales']:
                            if sc['key'] == s:
                                scale = sc
                                break
                        decode_params = {
                            'encoding': scale['encoding'],
                            'shape': scale['chunk_sizes'][0],
                            'dtype': info['data_type']
                        }
                        match_files(f1_abs, f2_abs, mode=mode, dtype=info['data_type'])
    else:
        for s in subs1:
            s1_abs = os.path.join(path1, s)
            s2_abs = os.path.join(path2, s)

            assert_folder_equality(s1_abs, s2_abs, parent=os.path.basename(path1))


def match_files(p1, p2, mode='json', dtype=None):
    if mode == 'json':
        with open(p1, 'r') as file1:
            d1 = json.load(file1)
        with open(p2, 'r') as file2:
            d2 = json.load(file2)
        ds1 = json.dumps(d1, sort_keys=True)
        ds2 = json.dumps(d2, sort_keys=True)

        assert ds1 == ds1
    else:
        with open(p1, 'rb') as f:
            d1 = f.read()
        with open(p2, 'rb') as f:
            d2 = f.read()

        if dtype == 'float32' and len(d1) % 4 != 0:
            #TODO: figure this case out
            return

        dec_d1 = np.frombuffer(bytearray(d1), dtype=dtype)
        dec_d2 = np.frombuffer(bytearray(d1), dtype=dtype)

        if mode == 'img':
            diff = abs(dec_d1 - dec_d2)
            if 'uint' in dtype:
                assert sum(diff > 5) == 0
            elif 'float' in dtype:
                # TODO: untested
                assert sum(diff > 0.01) == 0

        elif mode == 'mask':
            diff = abs(dec_d1 - dec_d2)
            if 'uint' in dtype:
                assert sum(diff > 0) == 0

        else:
            raise Exception("Unsupported mode")


