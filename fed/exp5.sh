
B_VALUES=("test5-q")  # 可能的字符串值
A_VALUES=("3e-2" "8e-2" "2e-1")  # 可能的字符串值
C_VALUES=("10" "20" "30" "40" "50" "60" "70" "80" "90" "100")  # 可能的整数值
b=${B_VALUES[0]}
echo "$b"> $b.txt
for ((i=0; i<${#A_VALUES[@]}; i++)); do
  for ((j=0; j<${#C_VALUES[@]}; j++)); do
    a=${A_VALUES[i]}
    c=${C_VALUES[j]}
    echo "$a + $c">> $b.txt
    python3 main_fed.py -dataset kdd -model dnn -save_model 0 -load_model 0 -local_bs 1000 -test_num -1 -attack 0 -defense 1 -alpha 1 -defense_lr $a -defense_epochs $c -num_users 1 -lr 1.5e-2 -decay 0.9 -epochs 300 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset kdd -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 1 -defense_lr $a -defense_epochs $c -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
  done
done