
  for rho in 99.5
  do
  
  for seed in 0 5 2
  do
    python main_aracnet.py \
      --dataset BFFHQ \
      --epochs_base_model 50 \
      --epochs_debiasing 70 \
      --seed $seed \
      --rho $rho \
      #--train_body_aracnet  

 done
 done     
 