
B_VALUES=("test4-q")  # 可能的字符串值
A_VALUES=("kdd" "mirai" "cic2017" "unsw")  # 可能的字符串值
C_VALUES=(5)  # 可能的整数值
b=${B_VALUES[0]}
echo "$b"> $b.txt
for ((i=0; i<${#A_VALUES[@]}; i++)); do
  for ((j=0; j<${#B_VALUES[@]}; j++)); do
    a=${A_VALUES[i]}
    b=${B_VALUES[j]}
    echo "dataset: $a">> $b.txt
    python3 test_kitsune.py -dataset $a -defense 0 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 1 -alpha 1 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 1 -alpha 0.5 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 1 -alpha 0.25 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 2 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 3 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 4 >> $b.txt
    python3 test_kitsune.py -dataset $a -defense 5 >> $b.txt

    python3 test_dnn.py -dataset $a -defense 0 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 1 -alpha 1 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 1 -alpha 0.5 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 1 -alpha 0.25 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 2 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 3 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 4 >> $b.txt
    python3 test_dnn.py -dataset $a -defense 5 >> $b.txt
  done
done