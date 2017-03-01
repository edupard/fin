import argparse
import os
import sys
import shlex
from config import get_config

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-n', '--dry-run', action='store_true',
                    help="Print out commands rather than executing them")

def new_cmd(session, name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex.quote(str(v)) for v in cmd)
    return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex.quote(cmd))


def create_commands(session, num_workers, shell='bash'):
    conda_cmd = ['source', 'activate', 'rl_fin']
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--num-workers', str(num_workers)]

    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job-name", "ps"])]
    conda_cmds_map = [new_cmd(session, "ps", conda_cmd)]
    for i in range(num_workers):
        cmds_map += [new_cmd(session,
            "w-%d" % i, base_cmd + ["--job-name", "worker", "--task", str(i)])]
        conda_cmds_map += [new_cmd(session, "w-%d" % i, conda_cmd)]

    conda_cmds_map += [new_cmd(session, "tb", conda_cmd)]
    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", get_config().log_dir, "--port", "12345"])]
    cmds_map += [new_cmd(session, "htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(get_config().log_dir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex.quote(arg) for arg in sys.argv if arg != '-n']), get_config().log_dir),
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


def run():
    args = parser.parse_args()
    cmds, notes = create_commands("a3c", args.num_workers)
    if args.dry_run:
        print("Dry-run mode due to -n flag, otherwise the following commands would be executed:")
    else:
        print("Executing the following commands:")
    print("\n".join(cmds))
    print("")
    if not args.dry_run:
        os.environ["TMUX"] = ""
        os.system("\n".join(cmds))
    print('\n'.join(notes))


if __name__ == "__main__":
    run()
