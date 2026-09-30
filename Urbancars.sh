#!/usr/bin/env bash
set -e
for seed in 0 1 2; do
    python main_aracnet.py \
        --dataset UrbanCars \
        --epochs_base_model 50 \
        --epochs_debiasing 50 \
        --seed $seed \
        --rho 95 \
        --train_body_aracnet
done
