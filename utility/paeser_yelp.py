import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN")

    parser.add_argument("--mask_rate", type=float, default=0.6, help="mask rate")

    parser.add_argument("--rand_ed", default="0.5", help="random edge drop rate")

    parser.add_argument('--info', type=float, default=0.01, help='info')

    parser.add_argument('--tem', type=float, default=0.1, help='temperature')

    parser.add_argument("--seed", type=int, default=2020, help="random seed for init")

    # parser.add_argument("--dataset", nargs="?", default="Amazon", help="[doubanmovie,movielens,Yelp2,lastfm,yelp2018]")
    parser.add_argument("--dataset", nargs="?", default="yelp", help="[doubanmovie,movielens,Yelp2,lastfm,yelp2018]")

    # parser.add_argument("--dataset_list", nargs="?",
    #                     default="Amazon")
    parser.add_argument("--dataset_list", nargs="?",
                        default="movielens")

    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    parser.add_argument("--SGL", default="MF", help="Model to train")

    parser.add_argument("--model_list", nargs="?",
                        default="['lightGCN','GCCF', 'NGCF','lightGCN-R','GCMC', 'MF', 'FISM', 'GCN', 'simiGCN', 'CapsGCF']")

    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

    # parser.add_argument('--att_dim', type=int, default=64, help='attention layer dim')

    parser.add_argument("--layer_att", type=int, default=0, help='use layer_attention or not')

    parser.add_argument('--GCNLayer', type=int, default=3, help="the layer number of GCN")

    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]', help='Output sizes of every layer')

    parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')

    parser.add_argument('--dim', type=int, default=64, help='embedding size')

    parser.add_argument('--drop_ratio', type=float, default=0.5, help='l2 regularization weight')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')#0.001

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='decay')

    parser.add_argument('--dropout', type=float, default=0.7, help='dropout')

    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider node dropout or not")

    parser.add_argument("--mess_keep_prob", nargs='?', default='[0.1, 0.1, 0.1]', help="ratio of node dropout")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument('--topK', nargs='?', default='[20]', help='size of Top-K')

    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}')

    parser.add_argument("--verbose", type=int, default=10, help="Test interval")

    parser.add_argument("--multicore", type=int, default=0, help="use multiprocessing or not in test")

    parser.add_argument("--save", type=bool, default=False, help="save SGL or not")

    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for SGL")

    parser.add_argument("--sparsity", type=bool, default=True, help="save SGL or not")
    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')

    parser.add_argument('--in_size', type=int, default=128, help='input size')
    parser.add_argument('--out_size', type=int, default=128, help='output size')


    parser.add_argument('--user', type=str, default="user")
    parser.add_argument('--item', type=str, default="item")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--type_num', type=int, default=[16239, 14284, 511, 47], help="[943, 1682, 18], [6170, 2753, 334, 22, 3857]")
    parser.add_argument('--hidden_dim', type=int, default=128)#512
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mp2vec_feat_drop", type=float, default=.2, help="input feature dropout")
    parser.add_argument("--feat_drop", type=float, default=0.5,
                        help="input feature dropout")#.2
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--feat_mask_rate", type=str, default="0.5,0.005,0.8",
                        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                            If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                            If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""")
    parser.add_argument("--encoder", type=str, default="han")
    parser.add_argument("--decoder", type=str, default="han")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--alpha_l", type=float, default=2, help="pow index for sce loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_gamma", type=float, default=0.99,
                        help="decay the lr by gamma for ExponentialLR scheduler")
    parser.add_argument("--use_mp2vec_feat_pred", action="store_true",
                        help="Set to True to use the mp2vec feature regularization.")
    parser.add_argument("--use_mp_edge_recon", action="store_true", default=True,
                        help="Set to True to use the meta-path edge reconstruction.")
    parser.add_argument("--mp_edge_mask_rate", type=str, default="0.5,0.005,0.8",
                        help="""The mask rate. If provide a float like '0.5', mask rate is static over the training. 
                            If provide two number connected by '~' like '0.4~0.9', mask rate is uniformly sampled over the training.
                            If Provide '0.7,-0.1,0.5', mask rate starts from 0.7, ends at 0.5 and reduce 0.1 for each epoch.""")





    return parser.parse_args()
