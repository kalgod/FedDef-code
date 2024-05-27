
B_VALUES=("test3-q")  # 可能的字符串值
A_VALUES=("kdd" "mirai" "cic2017" "unsw")  # 可能的字符串值
C_VALUES=(5)  # 可能的整数值
b=${B_VALUES[0]}
echo "$b"> $b.txt
for ((i=0; i<${#A_VALUES[@]}; i++)); do
  for ((j=0; j<${#B_VALUES[@]}; j++)); do
    a=${A_VALUES[i]}
    b=${B_VALUES[j]}
    echo "dataset: $a">> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 0 -alpha 1 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 1 -alpha 1 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 1 -alpha 0.5 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 1 -alpha 0.25 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 2 -alpha 1 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 3 -alpha 1 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 4 -alpha 1 -dataset $a -n_epochs 300 >> $b.txt
    python3 gan.py -pretrain 0 -use_ori 0 -defense 5 -alpha 1 -dataset $a -n_epochs 300 >> $b.txt
  done
done