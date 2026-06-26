
  for rho in 98
  do

  for seed in 1 0 8 10 11
  do
  
    python3 main_aracnet.py \
      --dataset waterbirds \
      --epochs_base_model 50 \
      --epochs_debiasing 50 \
      --seed $seed \
      --rho 95 \
      #--train_body_aracnet 
done
done

