#!/bin/bash
#SBATCH -J 1_3B
#SBATCH -p gpu-debug
#SBATCH -A r00114
#SBATCH -o 1_3B-baseline-8gpu_%j.txt
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240g
#SBATCH --time=00:20:00

# load cuda 11.7
mamba init
source ~/.bashrc
mamba activate megatron-TE

# UPDATE IT FOR TRAINING
NNODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export BASE_DIR=/N/slate/jindjia/bash_scripts/Course/hpc-course/Project
set -x

HOSTNAMES=$(scontrol show hostnames | sort -u)
HOSTLIST=""
FIRST_HOST=""
for HOST in $HOSTNAMES; do
  if [ -z "$FIRST_HOST" ]; then
    FIRST_HOST=$HOST
  fi
  HOST_ARRAY+=($HOST)
  HOSTLIST="${HOSTLIST}${HOST},"
done
HOSTLIST=${HOSTLIST%,}
MASTER_ADDR=$FIRST_HOST

srun --nodes=$NNODES --gres=gpu:$GPUS_PER_NODE --export=ALL,GPUS_PER_NODE=$GPUS_PER_NODE,MASTER_ADDR=$MASTER_ADDR,NNODES=$NNODES bash -c "./run_dist.sh"
