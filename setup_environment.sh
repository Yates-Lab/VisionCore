#!/bin/bash

###############################################################################
# 0  START FRESH  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
###############################################################################
conda deactivate 2>/dev/null || true
conda env remove -n yatesfv 2>/dev/null || true              # ignore if not present
conda create -n yatesfv python=3.12
conda activate yatesfv

conda install -y -c conda-forge
conda install git

# expose packages used by CUDA
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH"
export TRITON_LIBCUDA_PATH="$CUDA_HOME/lib64/stubs"

# verify compiler (i.e., fail before you try)
nvcc --version
which g++

###############################################################################
# ALL OTHER PACKAGES MANAGED BY PIP
echo "Installing packages from requirements.txt..."
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Installation complete."

pip install NeuroTools
git clone https://github.com/bicv/SLIP.git
cd SLIP
python setup.py install
cd ..
rm -rf SLIP

git clone https://github.com/bicv/LogGabor.git
cd LogGabor
python setup.py install
cd ..
rm -rf LogGabor


# install local packages
python -m pip install -e . -vvv

###############################################################################
# ENSSURE LCUDA IS VISIBLE FOR TORCH.COMPILE
# create a lib64 directory and drop a symlink the linker will see
mkdir -p $HOME/.local/lib          # or another dir you control
[ -f /lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -e $HOME/.local/lib/libcuda.so ] && ln -s /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/.local/lib/libcuda.so

export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# Now you can build cuda-dependent packages if needed