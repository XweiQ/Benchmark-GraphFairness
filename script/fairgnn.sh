declare -a devices=("2" "3" "4" "5" "6" "7")
declare -a dataset=('pokec_n')
for a in 0.001 0.01
do
for b in 5 20
do
for lr in 0.01
do
for p in 0.5
do 
for seed in {1..5}
do 
    device=${devices[$RANDOM % ${#devices[@]}]}
    python baseline_fairGNN.py \
        --model=fairgcn \
        --dataset $dataset \
        --epochs=1000 \
        --lr $lr \
        --weight_decay=1e-5 \
        --num_hidden=32 \
        --dropout $p \
        --alpha $a \
        --beta $b \
        --seed $seed \
        --device $device \
        --task train &
done
done
done
done
done


for a in 0.001 0.01
do
for b in 5 20
do
for lr in 0.01
do
for p in 0.5
do 
for seed in {1..5}
do 
    python baseline_fairGNN.py \
        --model=fairgcn \
        --dataset nba \
        --epochs=1000 \
        --lr $lr \
        --weight_decay=1e-5 \
        --num_hidden=32 \
        --dropout $p \
        --alpha $a \
        --beta $b \
        --seed $seed \
        --task eva
done
done
done
done
done

#!/bin/bash

# declare -a datasets=("MUTAG" "PROTEINS" "COLLAB" "IMDB-BINARY" "REDDIT-BINARY" "NCI1" "DD" "REDDIT-MULTI-5K")
# declare -a devices=("0" "1" "2" "3")

for dataset in "${datasets[@]}"
do
    for seed in {1..5}
    do
        device=${devices[$RANDOM % ${#devices[@]}]}
        echo "Running on dataset $dataset with device $device using seed $seed"
        CUDA_VISIBLE_DEVICES=$device python unsupervised.py --dataset_name $dataset --seed $seed > "results_new/${dataset}_${device}_seed${seed}.txt" &
    done
done
wait

declare -a devices=("1" "2" "3")
for seed in {1..5}
do 
    device=${devices[$RANDOM % ${#devices[@]}]}
    python baseline_fairGNN.py \
        --model=fairgcn \
        --dataset pokec_n \
        --epochs=1800 \
        --lr 1e-3 \
        --weight_decay=1e-5 \
        --num_hidden=128 \
        --dropout 0.5 \
        --alpha=50 \
        --beta=1 \
        --seed $seed \
        --device $device \
        --task train &
done

for seed in {1..5}
do 
    python baseline_fairGNN.py \
        --model=fairgcn \
        --dataset pokec_n \
        --epochs=1800 \
        --lr 1e-3 \
        --weight_decay=1e-5 \
        --num_hidden=128 \
        --dropout 0.5 \
        --alpha=50 \
        --beta=1 \
        --seed $seed \
        --task eva
done

python baseline_fairGNN.py \
    --model=fairgcn \
    --dataset nba \
    --epochs=1500 \
    --lr 1e-3 \
    --weight_decay=1e-5 \
    --num_hidden=128 \
    --dropout 0.5 \
    --alpha=10 \
    --beta=1 \
    --seed 1 \
    --device 1 \
    --task train
