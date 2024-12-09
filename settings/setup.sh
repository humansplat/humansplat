PROJ_DIR=$(pwd)

# install python packages
pip3 install -r requirements.txt 

# A modified gaussian splatting (+ alpha, depth, normal rendering)
if [ ! -d "extensions/diff-gaussian-rasterization" ]; then
    cd extensions && git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization 
fi
cd $PROJ_DIR && pip3 install ./extensions/diff-gaussian-rasterization




# download pre-trained ckpts

