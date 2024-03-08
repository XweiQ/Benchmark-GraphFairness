for data in 'pokec_z' 'pokec_n' 'nba'
do
python baseline_gnn.py \
    --drop_edge_rate_1=0.001 \
    --drop_edge_rate_2=0.001 \
    --drop_feature_rate_1=0.1 \
    --drop_feature_rate_2=0.1 \
    --epochs=2000 \
    --hidden=16 \
    --model=gcn \
    --dataset $data \
    --lr=1e-3 \
    --weight_decay=1e-4 \
    --dropout-0.5 \
    --num_layers=1 \
    --seed=1 \
    --device=1 \
    --task train 
done

python baseline_gnn.py \
    --drop_edge_rate_1=0.001 \
    --drop_edge_rate_2=0.001 \
    --drop_feature_rate_1=0.1 \
    --drop_feature_rate_2=0.1 \
    --epochs=2000 \
    --hidden=16 \
    --model=gcn \
    --dataset german \
    --lr=1e-3 \
    --weight_decay=1e-4 \
    --dropout-0.5 \
    --num_layers=1 \
    --seed=1 \
    --device=1 \
    --task train