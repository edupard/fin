#!/usr/bin/env python
import os
import shutil
import shlex
import sys
import multiprocessing
import time
import subprocess
import argparse
from enum import Enum

from config import get_config, Mode, parse_mode
from data_source.data_source import get_datasource

train_min = 1200
log_min = 2.5

start_seed_idx = 1
stop_seed_idx = 1


def is_widows_os():
    if os.name == 'nt':
        return True
    return False


num_workers = 64
if is_widows_os():
    num_workers = None
if num_workers is None:
    num_workers = multiprocessing.cpu_count()


# create tuple [window]->tmux send-keys comand to send [cmd] to [window] in tmux [session]
def wrap_cmd_for_tmux(session, window, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex.quote(str(v)) for v in cmd)
    return window, "tmux send-keys -t {}:{} {} Enter".format(session, window, shlex.quote(cmd))


def create_train_shell_commands(session, num_workers, train_seed, costs, mode, shell='bash'):
    conda_cmd = ['source', 'activate', 'rl_fin']
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable,
        'worker.py',
        '--num-workers', str(num_workers)]

    args = ["--job-name", "ps"]
    if costs:
        args += ["--costs"]
    args += ["--mode"]
    if mode == Mode.TRAIN:
        args += ["train"]
    if mode == Mode.CV:
        args += ["cv"]
    if mode == Mode.LOG:
        args += ["log"]
    cmds_map = [wrap_cmd_for_tmux(session, "ps", base_cmd + args)]
    conda_cmds_map = [wrap_cmd_for_tmux(session, "ps", conda_cmd)]
    for i in range(num_workers):
        args = ["--job-name", "worker",
                "--task", str(i),
                "--seed", str(train_seed)]
        if costs:
            args += ["--costs"]
        args += ["--mode"]
        if mode == Mode.TRAIN:
            args += ["train"]
        if mode == Mode.CV:
            args += ["cv"]
        if mode == Mode.LOG:
            args += ["log"]

        cmds_map += [wrap_cmd_for_tmux(session,
                                       "w-%d" % i, base_cmd + args)]
        conda_cmds_map += [wrap_cmd_for_tmux(session, "w-%d" % i, conda_cmd)]

    # tensorboard
    if mode != Mode.LOG:
        model_path = get_config().get_model_path(train_seed, costs, mode)
        conda_cmds_map += [wrap_cmd_for_tmux(session, "tb", conda_cmd)]
        cmds_map += [wrap_cmd_for_tmux(session, "tb", ["tensorboard", "--logdir", model_path, "--port", "12345"])]
    # htop
    cmds_map += [wrap_cmd_for_tmux(session, "htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(get_config().log_dir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex.quote(arg) for arg in sys.argv if arg != '-n']),
                                        get_config().log_dir),
    ]

    notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
    notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    notes += ["Point your browser to http://localhost:12345 to see Tensorboard"]

    cmds += [
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
    cmds += ["sleep 1"]

    for window, cmd in conda_cmds_map:
        cmds += [cmd]

    first = True
    for window, cmd in cmds_map:
        cmds += [cmd]
        if first:
            cmds += ["sleep 5"]
            first = False

    return cmds, notes


def create_validation_shell_commands(session, shell='bash'):
    conda_cmd = ['source', 'activate', 'rl_fin']
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--num-workers', '1']

    cmds_map = [wrap_cmd_for_tmux(session, "ps", base_cmd + ["--job-name", "ps"])]
    conda_cmds_map = [wrap_cmd_for_tmux(session, "ps", conda_cmd)]
    cmds_map += [wrap_cmd_for_tmux(session,
                                   "w-0", base_cmd + ["--job-name", "worker", "--task", "0"])]
    conda_cmds_map += [wrap_cmd_for_tmux(session, "w-0", conda_cmd)]

    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(get_config().log_dir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex.quote(arg) for arg in sys.argv if arg != '-n']),
                                        get_config().log_dir),
    ]

    notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
    notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]

    cmds += [
        "tmux kill-session -t {}".format(session),
        "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
    cmds += ["sleep 1"]

    for window, cmd in conda_cmds_map:
        cmds += [cmd]

    first = True
    for window, cmd in cmds_map:
        cmds += [cmd]
        if first:
            cmds += ["sleep 5"]
            first = False

    return cmds, notes


def start_nt_processes(num_workers, train_seed, costs, mode):
    processes = []
    workers = []
    args = [sys.executable, "worker.py", "--job-name", "ps", "--num-workers", str(num_workers)]
    if costs:
        args += ["--costs"]
    args += ["--mode"]
    if mode == Mode.TRAIN:
        args += ["train"]
    if mode == Mode.CV:
        args += ["cv"]
    if mode == Mode.LOG:
        args += ["log"]
    proc = subprocess.Popen(args)
    processes.append(proc)
    time.sleep(5)
    for idx in range(num_workers):
        args = [sys.executable,
                "worker.py",
                "--task", str(idx),
                "--num-workers", str(num_workers),
                "--seed", str(train_seed)
                ]
        if costs:
            args += ["--costs"]
        args += ["--mode"]
        if mode == Mode.TRAIN:
            args += ["train"]
        if mode == Mode.CV:
            args += ["cv"]
        if mode == Mode.LOG:
            args += ["log"]
        proc = subprocess.Popen(args)
        processes.append(proc)
        workers.append(proc)
    # tensorboard
    if mode != Mode.LOG:
        model_path = get_config().get_model_path(train_seed, costs, mode)
        proc = subprocess.Popen(["tensorboard.exe",
                                 "--logdir", model_path,
                                 "--port", "12345"])
        processes.append(proc)

    return processes, workers


def stop_nt_processes(processes):
    for proc in processes:
        proc.kill()


def copy_model(path, prev_path):
    if os.path.exists(path):
        print("Removing model %s" % path)
        shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(prev_path):
        print("Copying model %s to %s" % (prev_path, path))
        shutil.copytree(prev_path, path)
    else:
        print("Can't copy model %s to %s" % (prev_path, path))


def run_model(s_process, min, num_workers, train_seed, costs, mode):
    if is_widows_os():
        processes, workers = start_nt_processes(num_workers, train_seed, costs, mode)
    else:
        cmds, notes = create_train_shell_commands("a3c", num_workers, train_seed, costs, mode)
        os.environ["TMUX"] = ""
        os.system("\n".join(cmds))

    if mode == Mode.LOG and is_widows_os():
        print("Waiting for model to log")
        for w in workers:
            w.wait()
    else:
        print("Waiting %d min for model to %s" % (min, s_process))
        for idx in range(round(min)):
            time.sleep(60)
            min_passed = idx + 1
            print("%d min passed" % min_passed)
    print("Stopping %s process" % s_process)
    if is_widows_os():
        stop_nt_processes(processes)
    else:
        os.system("tmux kill-session -t a3c")
    time.sleep(5)


def main(mode, costs, copy_weights):
    data = get_datasource()
    data_length = data.shape[0]
    min_seed_idx = start_seed_idx
    max_seed = stop_seed_idx + 1
    if stop_seed_idx is None:
        max_seed = (data_length - get_config().ww) // get_config().retrain_interval
    for train_seed in range(min_seed_idx, max_seed):
        if train_seed < start_seed_idx:
            continue

        if mode == Mode.TRAIN:
            s_process = "train"
        elif mode == Mode.CV:
            s_process = "cv"
        elif mode == Mode.LOG:
            s_process = "log"
        print("Starting %s at %d seed" % (s_process, train_seed))
        if mode == Mode.TRAIN:
            if copy_weights:
                model_path = get_config().get_model_path(train_seed, costs, Mode.TRAIN)
                if costs:
                    prev_model_path = get_config().get_model_path(train_seed, False, Mode.TRAIN)
                    copy_model(model_path, prev_model_path)
                else:
                    if train_seed > 0:
                        prev_model_path = get_config().get_model_path(train_seed - 1, False, Mode.TRAIN)
                        copy_model(model_path, prev_model_path)
            run_model("train", train_min, num_workers, train_seed, costs, mode)
        # always copy weights during cv
        if mode == Mode.CV:
            model_path = get_config().get_model_path(train_seed, costs, Mode.TRAIN)
            cv_model_path = get_config().get_model_path(train_seed, costs, Mode.CV)
            copy_model(cv_model_path, model_path)
            run_model("cv", train_min, num_workers, train_seed, costs, mode)
        if mode == Mode.LOG:
            run_model("log", log_min, 1, train_seed, costs, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--mode', default="train", help="Network mode")
    parser.add_argument('--costs', action='store_true', help="Model include costs")
    parser.add_argument('--copy', action='store_true', help="Replace weights")

    args = parser.parse_args()
    mode = parse_mode(args.mode)
    main(mode, args.costs, args.copy)
