declare -a devices=("1" "2" "3")
for seed in {1..5}
do
    device=${devices[$RANDOM % ${#devices[@]}]}
    python baseline_gnn.py \
        --drop_edge_rate_1=0.001 \
        --drop_edge_rate_2=0.001 \
        --drop_feature_rate_1=0.1 \
        --drop_feature_rate_2=0.1 \
        --epochs=500 \
        --hidden=16 \
        --model=gcn \
        --dataset creditA \
        --lr 1e-2 \
        --weight_decay 1e-4 \
        --dropout 0.5 \
        --num_layers 2 \
        --seed $seed \
        --device $device \
        --task train &
done

for seed in {1..5}
do
    python baseline_gnn.py \
        --drop_edge_rate_1=0.001 \
        --drop_edge_rate_2=0.001 \
        --drop_feature_rate_1=0.1 \
        --drop_feature_rate_2=0.1 \
        --epochs=2000 \
        --hidden=16 \
        --model=gcn \
        --dataset creditA \
        --lr 1e-2 \
        --weight_decay 1e-4 \
        --dropout 0.5 \
        --num_layers 2 \
        --seed $seed \
        --task eva
done

python baseline_gnn.py \
    --drop_edge_rate_1=0.001 \
    --drop_edge_rate_2=0.001 \
    --drop_feature_rate_1=0.1 \
    --drop_feature_rate_2=0.1 \
    --epochs=1000 \
    --hidden=16 \
    --model=gcn \
    --dataset german \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_layers 1 \
    --seed 1 \
    --device 1 \
    --task train

python baseline_gnn.py \
    --drop_edge_rate_1=0.001 \
    --drop_edge_rate_2=0.001 \
    --drop_feature_rate_1=0.1 \
    --drop_feature_rate_2=0.1 \
    --epochs=500 \
    --hidden=16 \
    --model=gcn \
    --dataset creditA \
    --lr 1e-2 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_layers 2 \
    --seed 2 \
    --device 3 \
    --task train

python baseline_gnn.py \
    --drop_edge_rate_1=0.001 \
    --drop_edge_rate_2=0.001 \
    --drop_feature_rate_1=0.1 \
    --drop_feature_rate_2=0.1 \
    --epochs=1000 \
    --hidden=16 \
    --model=gcn \
    --dataset bail \
    --lr 1e-2 \
    --weight_decay 1e-5 \
    --dropout 0.5 \
    --num_layers 2 \
    --seed 2 \
    --device 2 \
    --task train


declare -a devices=("1" "2" "3")
for layer in 2 3
do
for lr in 1e-2 1e-3
do 
for seed in {1..5}
do
    device=${devices[$RANDOM % ${#devices[@]}]}
    python baseline_gnn.py \
        --drop_edge_rate_1=0.001 \
        --drop_edge_rate_2=0.001 \
        --drop_feature_rate_1=0.1 \
        --drop_feature_rate_2=0.1 \
        --epochs=2000 \
        --hidden=16 \
        --model=gcn \
        --dataset occuaption \
        --lr $lr \
        --weight_decay 1e-4 \
        --dropout 0.5 \
        --num_layers $layer \
        --seed $seed \
        --device $device \
        --task train &
done
done
done
for layer in 1 2 3
do
for lr in 1e-2 1e-3
do 
for seed in {1..5}
do
    python baseline_gnn.py \
        --drop_edge_rate_1=0.001 \
        --drop_edge_rate_2=0.001 \
        --drop_feature_rate_1=0.1 \
        --drop_feature_rate_2=0.1 \
        --epochs=2000 \
        --hidden=16 \
        --model=gcn \
        --dataset germanA \
        --lr $lr \
        --weight_decay 1e-4 \
        --dropout 0.5 \
        --num_layers $layer \
        --seed $seed \
        --task eva
done
done
done

declare -a devices=("2" "3" "4" "5" "6" "7")
for layer in 2 3 4 5
do
for lr in 1e-2 1e-3
do
for wd in 1e-4
do
for p in 0 0.5
do 
for seed in {1..5}
do
    device=${devices[$RANDOM % ${#devices[@]}]}
    python baseline_gnn.py \
        --drop_edge_rate_1=0.001 \
        --drop_edge_rate_2=0.001 \
        --drop_feature_rate_1=0.1 \
        --drop_feature_rate_2=0.1 \
        --epochs=1500 \
        --hidden=16 \
        --model mlp \
        --dataset pokec_n \
        --lr $lr \
        --weight_decay $wd \
        --dropout $p \
        --num_layers $layer \
        --seed $seed \
        --device $device \
        --task train &
done
done
done
done
done

for layer in 2 3 4 5
do
for lr in 1e-2 1e-3
do
for wd in 1e-4
do
for p in 0 0.5
do 
for seed in {1..5}
do
    python baseline_gnn.py \
        --drop_edge_rate_1=0.001 \
        --drop_edge_rate_2=0.001 \
        --drop_feature_rate_1=0.1 \
        --drop_feature_rate_2=0.1 \
        --epochs=1500 \
        --hidden=16 \
        --model mlp \
        --dataset pokec_z \
        --lr $lr \
        --weight_decay $wd \
        --dropout $p \
        --num_layers $layer \
        --seed $seed \
        --device $device \
        --task eva
done
done
done
done
done