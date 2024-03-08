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
    --dataset german \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_layers 1 \
    --seed $seed \
    --device 1 \
    --task train 
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
    --dataset german \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_layers 1 \
    --seed $seed \
    --device 1 \
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
    --model=gcn \
    --dataset bail \
    --lr 1e-2 \
    --weight_decay 1e-5 \
    --dropout 0 \
    --num_layers 2 \
    --seed $seed \
    --device 2 \
    --task train 
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
    --dataset bail \
    --lr 1e-2 \
    --weight_decay 1e-5 \
    --dropout 0 \
    --num_layers 2 \
    --seed $seed \
    --device 2 \
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
    --model=gcn \
    --dataset credit \
    --lr 1e-2 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_layers 3 \
    --seed $seed \
    --device 3 \
    --task train
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
    --dataset credit \
    --lr 1e-2 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --num_layers 3 \
    --seed $seed \
    --device 3 \
    --task eva 
done


# for lr in 0.01 0.001 0.0001
# do
#     for layers in 2 3
#     do
#         for seed in {1..5}
#         do
#             python baseline_gnn.py \
#                 --drop_edge_rate_1=0.001 \
#                 --drop_edge_rate_2=0.001 \
#                 --drop_feature_rate_1=0.1 \
#                 --drop_feature_rate_2=0.1 \
#                 --epochs=2000 \
#                 --hidden=16 \
#                 --model=gcn \
#                 --dataset occupation \
#                 --lr $lr \
#                 --weight_decay 1e-5 \
#                 --dropout 0.5 \
#                 --num_layers $layers \
#                 --seed $seed \
#                 --device 2 \
#                 --task eva
#         done
#     done
# done

# for lr in 0.01 0.001 0.0001
# do
#     for layers in 2 5
#     do
#         for seed in {1..5}
#         do
#             python baseline_gnn.py \
#                 --drop_edge_rate_1=0.001 \
#                 --drop_edge_rate_2=0.001 \
#                 --drop_feature_rate_1=0.1 \
#                 --drop_feature_rate_2=0.1 \
#                 --epochs=2000 \
#                 --hidden=16 \
#                 --model=mlp \
#                 --dataset occupation \
#                 --lr $lr \
#                 --weight_decay 1e-5 \
#                 --dropout 0.5 \
#                 --num_layers $layers \
#                 --seed $seed \
#                 --device 2 \
#                 --task eva
#         done
#     done
# done