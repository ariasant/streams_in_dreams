#!/bin/bash

# Load modules needed for EXP installation
module purge
module load modules/2.4-20250724 cmake uv gcc openmpi hdf5 libtirpc eigen fftw git python ninja doxygen/1.13.2


# Download EXP
git clone --recursive https://github.com/EXP-code/EXP.git -b devel

# Build EXP
cd EXP
cmake -G Ninja -B build -DCMAKE_INSTALL_RPATH=$PWD/install/lib  \
 --install-prefix $PWD/install \
 -DCMAKE_BUILD_TYPE=Release \
 -DBUILD_DOCS=YES  \
 -DENABLE_PYEXP=YES \
 -DENABLE_NBODY=YES \
 -DENABLE_USER_ALL=YES \
 -DENABLE_UTILS=ON
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
git clone https://github.com/adrn/gala.git -b devel

# Build Gala
cd gala

# Export path to EXP libraries (needs to be set when the environment is activated)
export GALA_EXP_PREFIX=$EXP_path/install

uv pip install -ve .

# Run benchmark tests
uv sync --all-extras
uv run pytest tests/benchmarks -k TestEXPTimeInterpBenchmark --codspeed --durations=0
