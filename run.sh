tmux kill-session -t a3c
source activate rl_fin
git pull origin exp-beta
git checkout -b exp-beta
python train.py -w 32