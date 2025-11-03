#!/usr/bin/bash
#SBATCH --job-name=exp_tests
#SBATCH --output=/mnt/home/asante/ceph/parc/job-%j.txt
#SBATCH --error=/mnt/home/asante/ceph/parc/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

export PYTHONUNBUFFERED=TRUE

source ceph/environments/exp/bin/
python /mnt/home/asante/ceph/stream_sim_fa.py
