declare -a datasets=('nba' 'pokec_z' 'pokec_n' 'german' 'credit' 'bail' 'germanA' 'creditA' 'bailA' 'synthetic' 'syn-1' 'syn-2' 'sport' 'occupation')
declare -a devices=("0")

sim_coeff=(0.2 0.4 0.6 0.8)
learning_rate=(0.01 0.001 0.0001)
weight_decay=(0.0001 0.00001)
dropout=(0 0.5 0.8)

for dataset in "${datasets[@]}"
do 
    for sim in (0.2 0.4 0.6 0.8)
    do
        for lr in (0.01 0.001 0.0001)
        do 
            for wd in (0.0001 0.00001)
            do
                for p in (0 0.5 0.8)
                do
                    for seed in {1..5}
                    do
                        python train_nifty.py \
                            --drop_edge_rate_1 0.001 \
                            --drop_edge_rate_2 0.001 \
                            --drop_feature_rate_1 0.1 \
                            --drop_feature_rate_2 0.1 \
                            --sim_coeff $sim \
                            --lr $lr \
                            --weight_decay $wd \
                            --dropout $p \
                            --hidden 16 \
                            --epochs 1000 \
                            --model ssf \
                            --encoder gcn \
                            --dataset $dataset \
                            --seed $seed \
                            --device $device \
                            --task train 
                done
            done
        done
    done
done

for dataset in "${datasets[@]}"
do 
    for sim in (0.2 0.4 0.6 0.8)
    do
        for lr in (0.01 0.001 0.0001)
        do 
            for wd in (0.0001 0.00001)
            do
                for p in (0 0.5 0.8)
                do
                    for seed in {1..5}
                    do
                        python train_nifty.py \
                            --drop_edge_rate_1 0.001 \
                            --drop_edge_rate_2 0.001 \
                            --drop_feature_rate_1 0.1 \
                            --drop_feature_rate_2 0.1 \
                            --sim_coeff $sim \
                            --lr $lr \
                            --weight_decay $wd \
                            --dropout $p \
                            --hidden 16 \
                            --epochs 1000 \
                            --model ssf \
                            --encoder gcn \
                            --dataset $dataset \
                            --seed $seed \
                            --device $device \
                            --task eva
                done
            done
        done
    done
done