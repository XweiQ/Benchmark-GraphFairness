declare -a devices=("1" "2" "3")
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
        --model=mlp \
        --dataset synthetic \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --dropout 0.5 \
        --num_layers 2 \
        --yscale 0.5 \
        --sscale 0.5 \
        --covy 10 \
        --covs 10 \
        --device $device \
        --seed $seed \
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
        --model=mlp \
        --dataset synthetic \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --dropout 0.5 \
        --num_layers 2 \
        --seed $seed \
        --task eva
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
        --model=mlp \
        --dataset synthetic \
        --lr 1e-3 \
        --weight_decay 1e-5 \
        --dropout 0.5 \
        --num_layers 2 \
        --seed $seed \
        --yscale 0.5 \
        --sscale 0.5 \
        --covy 10 \
        --covs 10 \
        --prob_s0y0 0.25 \
        --prob_s0y1 0.25 \
        --prob_s1y0 0.25 \
        --prob_s1y1 0.25 \
        --pysame 0.006 \
        --pssame0 0.006 \
        --pssame1 0.03 \
        --pydif 0.002 \
        --psdif 0.002 \
        --device 0 \
        --task eva
done