
B_VALUES=("test2-q")  # 可能的字符串值
A_VALUES=("kdd" "mirai" "cic2017" "unsw")  # 可能的字符串值
C_VALUES=(5)  # 可能的整数值
b=${B_VALUES[0]}
echo "$b"> $b.txt
for ((i=0; i<${#A_VALUES[@]}; i++)); do
  for ((j=0; j<${#B_VALUES[@]}; j++)); do
    a=${A_VALUES[i]}
    b=${B_VALUES[j]}
    echo "dataset: $a">> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 0 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 0 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 0 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 0 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 1 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 1 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 0.5 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 1 -alpha 0.5 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 0.5 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 1 -alpha 0.5 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 0.25 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 1 -alpha 0.25 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 1 -alpha 0.25 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 1 -alpha 0.25 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 2 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 2 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 2 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 2 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 3 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 3 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 3 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 3 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 4 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 4 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 4 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 4 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt

    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 1 -defense 5 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 0 -local_bs 1 -test_num -1 -attack 2 -defense 5 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 1 -defense 5 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
    python3 main_fed.py -dataset $a -model dnn -save_model 0 -load_model 1 -local_bs 1 -test_num -1 -attack 2 -defense 5 -alpha 1 -defense_lr 2e-1 -defense_epochs 40 -num_users 1 -lr 1e-2 -decay 0.8 -epochs 5 -decay_epochs 20 -iid 1 >> $b.txt
  done
done