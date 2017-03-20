#!/usr/bin/env python
import os
import shutil
import shlex
import sys
import multiprocessing
import time
import subprocess
import argparse

from config import get_config
from data_source.data_source import get_datasource

start_seed = 0

train_min = 600
costs_train_min = 60
validation_min = 1.5

train_min_a = []
costs_train_min_a = []

start_seed = 0
stop_seed = 0

num_workers = 32
if num_workers is None:
    num_workers = multiprocessing.cpu_count()


# create tuple [window]->tmux send-keys comand to send [cmd] to [window] in tmux [session]
def wrap_cmd_for_tmux(session, window, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex.quote(str(v)) for v in cmd)
    return window, "tmux send-keys -t {}:{} {} Enter".format(session, window, shlex.quote(cmd))


def create_train_shell_commands(session, num_workers, model_path, shell='bash'):
    conda_cmd = ['source', 'activate', 'rl_fin']
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--num-workers', str(num_workers)]

    cmds_map = [wrap_cmd_for_tmux(session, "ps", base_cmd + ["--job-name", "ps"])]
    conda_cmds_map = [wrap_cmd_for_tmux(session, "ps", conda_cmd)]
    for i in range(num_workers):
        cmds_map += [wrap_cmd_for_tmux(session,
                                       "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i)])]
        conda_cmds_map += [wrap_cmd_for_tmux(session, "w-%d" % i, conda_cmd)]

    conda_cmds_map += [wrap_cmd_for_tmux(session, "tb", conda_cmd)]
    cmds_map += [wrap_cmd_for_tmux(session, "tb", ["tensorboard", "--logdir", model_path, "--port", "12345"])]
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


def start_nt_processes(num_workers, model_path):
    processes = []
    proc = subprocess.Popen([sys.executable, "worker.py", "--job-name", "ps", "--num-workers", str(num_workers)])
    processes.append(proc)
    time.sleep(5)
    for idx in range(num_workers):
        proc = subprocess.Popen([sys.executable, "worker.py", "--task", str(idx), "--num-workers", str(num_workers)])
        processes.append(proc)
    proc = subprocess.Popen(["tensorboard.exe", "--logdir", model_path, "--port", "12345"])
    processes.append(proc)
    return processes


def start_nt_validation_processes():
    processes = []
    proc = subprocess.Popen([sys.executable, "worker.py", "--job-name", "ps", "--num-workers", "1"])
    processes.append(proc)
    time.sleep(5)
    proc = subprocess.Popen([sys.executable, "worker.py", "--task", "0", "--num-workers", "1"])
    processes.append(proc)
    return processes


def stop_nt_processes(processes):
    for proc in processes:
        proc.kill()


def copy_model(path, prev_path):
    weights_path = os.path.join(path, 'train')
    prev_weights_path = os.path.join(prev_path, 'train')
    if os.path.exists(prev_weights_path):
        print("Copying model from %s to %s" % (prev_path, path))
        shutil.copytree(prev_weights_path, weights_path)


def prepare_config(evaluation, retrain_seed, costs_on):
    with open("nn.ini", "w") as text_file:
        text_file.writelines(['[DEFAULT]\n',
                              'evaluation = {}\n'.format('yes' if evaluation else 'no'),
                              'retrain_seed = {}\n'.format(retrain_seed),
                              'costs_on = {}\n'.format('yes' if costs_on else 'no')])


def train(num_workers, retrain_seed, costs_on, model_path):
    if os.name == 'nt':
        processes = start_nt_processes(num_workers, model_path)
    else:
        cmds, notes = create_train_shell_commands("a3c", num_workers, model_path)
        os.environ["TMUX"] = ""
        os.system("\n".join(cmds))

    if costs_on:
        wait_min = costs_train_min
        if retrain_seed < len(costs_train_min_a):
            wait_min = costs_train_min[retrain_seed]
    else:
        wait_min = train_min
        if retrain_seed < len(train_min_a):
            wait_min = train_min_a[retrain_seed]

    print("Waiting %d min for model to train" % wait_min)
    for idx in range(round(wait_min)):
        time.sleep(60)
        min_passed = idx + 1
        print("%d min passed" % min_passed)
    print("Stopping train process")
    if os.name == 'nt':
        stop_nt_processes(processes)
    else:
        os.system("tmux kill-session -t a3c")
    time.sleep(5)


def validate():
    if os.name == 'nt':
        processes = start_nt_validation_processes()
    else:
        cmds, notes = create_validation_shell_commands("a3c")
        os.environ["TMUX"] = ""
        os.system("\n".join(cmds))
    print("Waiting %.2f min for model to validate" % validation_min)
    time.sleep(60 * validation_min)
    print("Stopping validation process")
    if os.name == 'nt':
        stop_nt_processes(processes)
    else:
        os.system("tmux kill-session -t a3c")
    time.sleep(5)


def cross_validation(copy_weights, skip_train, skip_costs_train, skip_validation):
    data = get_datasource()
    data_length = data.shape[0]
    min_seed = start_seed
    max_seed = stop_seed
    if stop_seed is None:
        max_seed = (data_length - get_config().ww - get_config().train_length) // get_config().retrain_interval
    # train without costs
    if not skip_train:
        for retrain_seed in range(min_seed, max_seed + 1):
            if retrain_seed < start_seed:
                continue
            print("Starting train at %d seed step" % retrain_seed)
            model_path = get_config().get_model_path(retrain_seed, False)
            # remove model in dry_run mode
            # if dry_run:
            #     print("Dry run - removing model at %s" % model_path)
            #     shutil.rmtree(model_path, ignore_errors=True)
            # if no model - copy prev model
            if copy_weights and not os.path.exists(model_path) and retrain_seed > 0:
                prev_model_path = get_config().get_model_path(retrain_seed - 1, False)
                copy_model(model_path, prev_model_path)
            # train model
            # prepare ini file
            print("Preparing nn.ini file")
            prepare_config(False, retrain_seed, False)
            print("Start training")
            train(num_workers, retrain_seed, False, model_path)
    if not skip_costs_train:
        for retrain_seed in range(min_seed, max_seed + 1):
            if retrain_seed < start_seed:
                continue
            print("Starting train with costs at %d seed step" % retrain_seed)
            model_path = get_config().get_model_path(retrain_seed, True)
            # remove model if exists - we always learn model with costs using prev model without costs
            shutil.rmtree(model_path, ignore_errors=True)
            # copy model without costs
            prev_model_path = get_config().get_model_path(retrain_seed, False)
            copy_model(model_path, prev_model_path)
            # prepare ini file
            print("Preparing nn.ini file")
            prepare_config(False, retrain_seed, True)
            print("Start training")
            train(num_workers, retrain_seed, True, model_path)
    if not skip_validation:
        for retrain_seed in range(min_seed, max_seed + 1):
            if retrain_seed < start_seed:
                continue
            print("Starting validation at %d seed step" % retrain_seed)
            print("Preparing nn.ini file")
            prepare_config(True, retrain_seed, False)
            validate()
    if os.path.exists("nn.ini"):
        os.remove("nn.ini")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--copy-weights', action='store_true', help="Copy weights from previous model")
    parser.add_argument('--skip-train', action='store_true', help="Skip train step")
    parser.add_argument('--skip-costs-train', action='store_true', help="Skip train with costs step")
    parser.add_argument('--skip-validation', action='store_true', help="Skip validation step")
    args = parser.parse_args()
    cross_validation(args.copy_weights, args.skip_train, args.skip_costs_train, args.skip_validation)
