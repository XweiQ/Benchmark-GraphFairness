
declare -a devices=("1" "2" "3")
for sim in 0.4
do
for seed in {1..5}
do 
    device=${devices[$RANDOM % ${#devices[@]}]}
    python nifty_sota_gnn.py \
        --drop_edge_rate_1 0.001 \
        --drop_edge_rate_2 0.001 \
        --drop_feature_rate_1 0.1 \
        --drop_feature_rate_2 0.1 \
        --dropout 0.5 \
        --hidden 16 \
        --lr 1e-2 \
        --weight_decay 1e-5 \
        --epochs 1000 \
        --model ssf \
        --encoder gcn \
        --dataset nba \
        --sim_coeff $sim \
        --seed $seed \
        --device $device \
        --task train &
done
done

for sim in 0.4
do
for seed in {1..5}
do 
    python nifty_sota_gnn.py \
        --drop_edge_rate_1 0.001 \
        --drop_edge_rate_2 0.001 \
        --drop_feature_rate_1 0.1 \
        --drop_feature_rate_2 0.1 \
        --dropout 0.5 \
        --hidden 16 \
        --lr 1e-2 \
        --weight_decay 1e-5 \
        --epochs 1000 \
        --model ssf \
        --encoder gcn \
        --dataset nba \
        --sim_coeff $sim \
        --seed $seed \
        --device $device \
        --task eva
done
done

declare -a devices=("1" "2" "3")
for seed in {1..5}
do 
    device=${devices[$RANDOM % ${#devices[@]}]}
    python nifty_sota_gnn.py \
        --drop_edge_rate_1 0.001 \
        --drop_edge_rate_2 0.001 \
        --drop_feature_rate_1 0.1 \
        --drop_feature_rate_2 0.1 \
        --dropout 0.5 \
        --hidden 16 \
        --lr 1e-2 \
        --weight_decay 1e-5 \
        --epochs 1000 \
        --model ssf \
        --encoder gcn \
        --dataset nba \
        --sim_coeff 0.4 \
        --device $device \
        --seed $seed \
        --task train &
done