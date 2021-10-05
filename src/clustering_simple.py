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
from sklearn.cluster import KMeans
import networkx as nx

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--hops', type=int, default=20,
                    help='number of hops')
parser.add_argument('--dataset', type=str,
                    default='citeseer', help='type of dataset.')

args = parser.parse_args()


def run(args):
    print("Using {} dataset".format(args.dataset))

    if args.dataset == 'wiki':
        adj, features, y, _  = load_wiki()
    elif args.dataset in ['cora', 'citeseer', 'pubmed']:
        adj, features, y, _, _, _, _ = load_data(args.dataset)
    else:
        print("Dataset not supported!")
        exit()

    n_nodes, feat_dim = features.shape
    n_clusters = y.max()+1

    for hop in range(args.hops, args.hops+1):
        input_features = 0.
        adj_norm = normalize_adj(adj, r=0.5)
        features_list = []
        features_list.append(features)
        for _ in range(hop):
            features_list.append(torch.spmm(adj_norm, features_list[-1]))

        input_features = features_list[hop]

        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
        y_pred = kmeans.fit_predict(input_features.numpy())
        eva(y, y_pred, hop)


if __name__ == '__main__':
    run(args)
