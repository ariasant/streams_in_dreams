#!/bin/bash

# Load modules needed for EXP installation
module purge
module load modules/2.4-20250724 cmake uv gcc openmpi hdf5 libtirpc eigen fftw git python ninja doxygen/1.13.2


# Download EXP
git clone --recursive https://github.com/EXP-code/EXP.git

# Build EXP
cd EXP
cmake -G Ninja -B build -DCMAKE_INSTALL_RPATH=$PWD/install/lib --install-prefix $PWD/install -DBUILD_DOCS=YES -DENABLE_PYEXP=YES
cmake --build build
cmake --install build 

EXP_path=$PWD

# Create virtual environment
cd .. 
venv_name="expgala"


uv venv $venv_name

# Activate environment
source $venv_name/bin/activate

# Install packages from working venv
uv pip install -r /mnt/home/asante/ceph/environments/exp/requirements.txt

# Download gala development version
git clone https://github.com/adrn/gala.git

# Build Gala
cd gala
uv venv
# Export path to EXP libraries (needs to be set when the environment is activated)
export GALA_EXP_PREFIX=$EXP_path/
export GALA_EXP_LIB_PATH=$EXP_path/install/lib/
uv pip install -ve .

