#!/usr/bin/env bash
set -e
for rho in 99 95; do
    for seed in 1 0 5; do
        python main_aracnet.py \
            --dataset bar \
            --epochs_base_model 50 \
            --epochs_debiasing 50 \
            --seed $seed \
            --rho $rho \
            --train_body_aracnet
    done
done
