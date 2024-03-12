import argparse

def get_base_args(parser):
    # dataset setting
    parser.add_argument('--dataset', type=str, default='german')
    # Training settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', default=1, help='select gpu.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    # model setting
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'mlp'])
    parser.add_argument("--num_layers", type=int, default=2,
                            help="number of hidden layers")            
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    # train or evaluate
    parser.add_argument('--task', type=str, default='train', help='train models or evaluate')

    args = parser.parse_known_args()[0]
    return args

def get_nifty_args(parser):
    # dataset settings
    parser.add_argument('--dataset', type=str, default='occupation')
    # Training settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', default=0, help='select gpu.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    # model settings
    parser.add_argument('--model', type=str, default='ssf')
    parser.add_argument('--encoder', type=str, default='gcn')            
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--proj_hidden', type=int, default=16,
                        help='Number of hidden units in the projection layer of encoder.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,
                        help='drop edge for first augmented graph')
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                        help='drop edge for second augmented graph')
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.1,
                        help='drop feature for first augmented graph')
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.1,
                        help='drop feature for second augmented graph')
    parser.add_argument('--sim_coeff', type=float, default=0.5,
                        help='regularization coeff for the self-supervised task')
    parser.add_argument("--num_heads", type=int, default=1,
                            help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                            help="number of hidden layers")
    # train or evaluate
    parser.add_argument('--task', type=str, default='train', help='train the model or evaluate')
    
    args = parser.parse_known_args()[0]
    return args

def get_fairgnn_args(parser):
    # dataset settings
    parser.add_argument('--dataset', type=str, default='pokec_n')
    # training settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', default=1, help='select gpu.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    # model settings
    parser.add_argument('--model', type=str, default="fairgcn",
                        help='the type of model GCN/GAT')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units of the sensitive attribute estimator')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=4,
                        help='The hyperparameter of alpha')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
    parser.add_argument('--num-hidden', type=int, default=32,
                        help='Number of hidden units of classifier.')
    parser.add_argument("--num-heads", type=int, default=1,
                            help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--sens_number', type=int, default=200,
                        help="the number of sensitive attributes")
    parser.add_argument('--label_number', type=int, default=500,
                        help="the number of labels")
    parser.add_argument('--run', type=int, default=0,
                        help="kth run of the model")
    parser.add_argument('--pretrained', type=bool, default=False,
                        help="load a pretrained model")
    parser.add_argument('--task', type=str, default='train', help='train the model or evaluate')

    args = parser.parse_known_args()[0]
    return args