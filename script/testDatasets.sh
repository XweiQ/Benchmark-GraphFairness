for dataset in 'germanA' 'bailA' 'creditA' #'nba' 'pokec_z' 'pokec_n'
do 
    python baseline_gnn.py \
        --model=mlp \
        --dataset $dataset
done
