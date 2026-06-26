 
  for rho in 99 95 
  do
  
  for seed in 1 0 5
  do
    python main_aracnet.py \
      --dataset bar \
      --epochs_base_model 50 \
      --epochs_debiasing 50 \
      --seed $seed \
      --rho $rho \
      #--train_body_aracnet \

  
done
done



  for rho in 99.5 99 95 98 
  do
  
  for seed in 0 1 2 
  do
    python main_aracnet.py \
      --dataset BFFHQ \
      --epochs_base_model 50 \
      --epochs_debiasing 100 \
      --seed $seed \
      --rho $rho \
      #--train_body_aracnet  

 done
 done     





