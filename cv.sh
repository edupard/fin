#!/bin/bash

tmux kill-session -t cv
source activate rl_fin
rm -r -f fin
git clone https://github.com/edupard/fin.git
cd fin
git pull origin exp-1
git checkout -b exp-1

tmux new-session -s cv -n mgr -d bash
tmux send-keys -t cv:mgr 'source activate rl_fin' Enter
tmux send-keys -t cv:mgr 'python cv.py --mode train' Enter