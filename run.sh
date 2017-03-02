tmux kill-session -t a3c
source activate rl_fin
rm -r -f fin
git clone https://github.com/edupard/fin.git
git pull origin exp-1
cd fin
git checkout -b exp-1
python train.py -w 32