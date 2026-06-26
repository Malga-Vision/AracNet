for seed in 1 0 2 
do
for rho in  95 99 98 99.5
  do
    python main_aracnet.py \
      --dataset cifar10c \
      --epochs_base_model 50 \
      --epochs_debiasing 100 \
      --seed $seed \
      --rho $rho  \
      #--train_body_aracnet  
done
done

