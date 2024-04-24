#!/bin/bash

set -x

# Change for multinode config
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODE_RANK=$SLURM_NODEID

VOCAB_FILE=/N/scratch/jindjia/thepile/vocab.json
MERGE_FILE=/N/scratch/jindjia/thepile/merges.txt
DATA_PATH=/N/scratch/jindjia/thepile/pile_text_document
CHECKPOINT_PATH=/N/scratch/jindjia/checkpoint/Course/hpc-course/Project
WANDB_DIR=/tmp/"${SLURM_JOB_ID}"/wandb
TENSORBOARD_DIR=/tmp/"${SLURM_JOB_ID}"/tensorboard

copy_files() {
    mkdir -p "${BASE_DIR}/${SLURM_JOB_ID}/wandb"
    mkdir -p "${BASE_DIR}/${SLURM_JOB_ID}/tensorboard"
    cp -R "${WANDB_DIR}" "${BASE_DIR}/${SLURM_JOB_ID}/wandb"
    cp -R "${TENSORBOARD_DIR}" "${BASE_DIR}/${SLURM_JOB_ID}/tensorboard"
}
trap copy_files EXIT

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

# MODEL_ARGS="
#     --num-layers 24 \
#     --hidden-size 1024 \
#     --num-attention-heads 16 \
#     --seq-length 2048 \
#     --max-position-embeddings 2048 \
# "
MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

TRAINING_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters 80000 \
"

OPTIMIZER_ARGS="
    --lr 0.0003 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00003 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
    --fp16 \
    --use-distributed-optimizer \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-cache-path /tmp/${SLURM_JOB_ID}/data_cache \
    --distributed-storage \
"

OUTPUT_ARGS="
    --log-interval 10 \
    --timing-log-level 2 \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --tensorboard-log-interval 1 \
    --save-interval 5002 \
    --eval-interval 100 \
    --eval-iters 10 \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --wandb-project DEBUG \
    --wandb-save-dir ${WANDB_DIR} \
    --wandb-exp-name 350M-baseline \
"

QUANTIZE_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity selective \
"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /N/slate/jindjia/LLM/Megatron-LM-Final-Design

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $QUANTIZE_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --exit-duration-in-mins 2840