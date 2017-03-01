# update system
sudo apt-get update
sudo apt-get upgrade
#clone repo
git clone https://github.com/edupard/fin.git
#install anaconda
wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh
#create conda environment
conda create --name rl_fin python=3.5
source activate rl_fin
#install python packages
pip install gym==0.7.2
#dependencies for atari
sudo apt-get install cmake
sudo apt-get install zlib1g-dev
sudo apt-get install build-essential
pip install atari-py==0.0.18
#other packages
pip install numpy==1.12.0
pip install scipy==0.18.1
pip install Pillow==3.4.2
pip install opencv-python
pip install matplotlib==2.0.0
pip install tensorflow==0.12
#software rendering
pip install pygame
#headless opengl
sudo apt-get install mesa-utils
sudo apt-get install libglfw3
pip install glfw==1.3.3
pip install PyOpenGL==3.1.0

#install mesa utils
sudo apt-get install mesa-utils
#install x server
sudo apt-get install xorg openbox
#install dummy driver
//https://www.lxtreme.nl/blog/headless-x11/
sudo apt-get install xserver-xorg-video-dummy

#install virtual framebuffer
sudo apt-get install xvfb
export DISPLAY=:0
Xvfb :0 -screen 0 640x480x24 -fbdir /var/tmp&
Xvfb :0 -screen 0 1024x768x24 +extension RANDR &

glxgears