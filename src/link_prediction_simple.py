from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str,
                    default='cora', help='type of dataset.')
parser.add_argument('--hops', type=int, default=20, help='number of hops')

args = parser.parse_args()


def run(args):
    print("Using {} dataset".format(args.dataset))
    if args.dataset == 'wiki':
        adj, features, y, _  = load_wiki()
    else:
        adj, features, y, _, _, _, _ = load_data(args.dataset)
    n_nodes, feat_dim = features.shape

    adj_orig = adj
    adj_orig = adj_orig - \
        sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [
                      0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, _, _, _, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    for hop in range(args.hops, args.hops+1):
        input_features = 0.
        adj_norm = normalize_adj(adj, r=0.5)
        features_list = []
        features_list.append(features)
        for _ in range(hop):
            features_list.append(torch.spmm(adj_norm, features_list[-1]))

        input_features = features_list[hop]

        sim = torch.sigmoid(torch.mm(input_features, input_features.T))

        roc_score, ap_score = get_roc_score(sim.numpy(), adj_orig, test_edges, test_edges_false)
        print(f'AUC: {roc_score:.4f}, AP: {ap_score:.4f}, Hop: {hop:02d}')


if __name__ == '__main__':
    set_seed(args.seed)
    run(args)
