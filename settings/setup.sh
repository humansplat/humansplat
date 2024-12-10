PROJ_DIR=$(pwd)

# install python packages
pip3 install -r requirements.txt 

# A modified gaussian splatting (+ alpha, depth, normal rendering)
if [ ! -d "extensions/diff-gaussian-rasterization" ]; then
    cd extensions && git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization 
fi

# install gaussian-rasterization
cd $PROJ_DIR && pip3 install ./extensions/diff-gaussian-rasterization

# install pytorch3d
{
    python3 extensions/install_pytorch3d.py
} || {
    export MAKEFLAGS="-j32" pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
}

# download pre-trained ckpts





