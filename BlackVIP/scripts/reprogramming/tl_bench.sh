#!/bin/bash
cd ../..
DATA=/YOURPATH
TRAINER=BAR

SHOTS=16
CFG=vit_b16

DATASET=$1
ep=$2

init_lr=$3
min_lr=$4

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/shot${SHOTS}_ep${ep}/seed${SEED}

    CUDA_VISIBLE_DEVICES=7 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAIN.CHECKPOINT_FREQ 500 \
    DATASET.SUBSAMPLE_CLASSES all \
    OPTIM.MAX_EPOCH $ep \
    TRAINER.BAR.LRS [$init_lr,$min_lr] \
    INPUT.SIZE [194,194]
done