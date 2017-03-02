tmux kill-session -t a3c
source activate rl_fin
rm -r -f fin
git clone https://github.com/edupard/fin.git
git pull origin exp-0
git checkout -b exp-0
cd fin
python train.py -w 32