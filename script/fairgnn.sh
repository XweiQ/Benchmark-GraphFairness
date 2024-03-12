declare -a datasets=('nba' 'pokec_z' 'pokec_n' 'german' 'credit' 'bail' 'germanA' 'creditA' 'bailA' 'synthetic' 'syn-1' 'syn-2' 'sport' 'occupation')
declare -a devices=("0")

learning_rate=(0.01 0.001 0.0001)
weight_decay=(0.0001 0.00001)
dropout=(0 0.5 0.8)
alpha=(5 50 100)
beta=(1 5 20)

for dataset in "${datasets[@]}"
do 
    for lr in (0.01 0.001 0.0001)
    do 
        for wd in (0.0001 0.00001)
        do
            for p in (0 0.5 0.8)
            do
                for a in (5 50 100)
                do
                    for b in (1 5 20)
                    do
                        for seed in {1..5}
                        do
                            python train_fairGNN.py \
                                --model=fairgcn \
                                --dataset $dataset \
                                --epochs=1000 \
                                --lr $lr \
                                --weight_decay=$wd \
                                --num_hidden=32 \
                                --dropout $p \
                                --alpha $a \
                                --beta $b \
                                --seed $seed \
                                --device $device \
                                --task train
                        done
                    done
                done
            done
        done
    done
done

for dataset in "${datasets[@]}"
do 
    for lr in (0.01 0.001 0.0001)
    do 
        for wd in (0.0001 0.00001)
        do
            for p in (0 0.5 0.8)
            do
                for a in (5 50 100)
                do
                    for b in (1 5 20)
                    do
                        for seed in {1..5}
                        do
                            python train_fairGNN.py \
                                --model=fairgcn \
                                --dataset $dataset \
                                --epochs=1000 \
                                --lr $lr \
                                --weight_decay=$wd \
                                --num_hidden=32 \
                                --dropout $p \
                                --alpha $a \
                                --beta $b \
                                --seed $seed \
                                --device $device \
                                --task eva
                        done
                    done
                done
            done
        done
    done
done