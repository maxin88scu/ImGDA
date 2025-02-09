"""
command line arguments for ImGDA

"""

import argparse
def getargs():

    parser=argparse.ArgumentParser(description='ImGDA')

    parser.add_argument("--source",type=str,default='ACMv9',help="source domain")
    parser.add_argument("--target",type=str,default='Citationv1',help="target domain")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--imbalance_factor", type=int, default=20,help="imbalance ratio of source")
    parser.add_argument("--weight_decay", type=float, default=1e-3,help="weight decay rate for GNN")
    parser.add_argument("--drop_out", type=float, default=1e-1,help="drop out rate for GNN")
    parser.add_argument("--encoder_dim", type=int, default=512,help="dimension of encoder")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--epochs", type=int, default=200,help="training epoch")
    parser.add_argument("--dynamic_temperature", type=float, default=0.5,help="dynamic temperature")
    parser.add_argument("--k", type=int, default=200,help="number of anchor nodes")
    parser.add_argument("--times", type=int, default=5, help="number of training times")

    args=parser.parse_args()

    return args