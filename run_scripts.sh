#!/usr/bin/bash
#SBATCH --job-name=exp_tests
#SBATCH --output=/mnt/home/asante/jobs/job-%j.txt
#SBATCH --error=/mnt/home/asante/jobs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64

export PYTHONUNBUFFERED=TRUE

module load modules/2.4-20250724 cmake uv gcc openmpi hdf5 libtirpc eigen fftw git python ninja doxygen/1.13.2

source /mnt/home/asante/streams_in_dreams/nbody/bin/activate
python /mnt/home/asante/streams_in_dreams/nbody_sim.py
