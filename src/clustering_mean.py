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
        r_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for r in r_list:
            adj_norm = normalize_adj(adj, r)
            features_list = []
            features_list.append(features)
            for _ in range(hop):
                features_list.append(torch.spmm(adj_norm, features_list[-1]))

            weight_list = []
            norm_fea = torch.norm(features, 2, 1).add(1e-10)
            for fea in features_list:
                norm_cur = torch.norm(fea, 2, 1).add(1e-10)
                
                temp = torch.div((features*fea).sum(1), norm_fea)
                temp = torch.div(temp, norm_cur)
                weight_list.append(temp.unsqueeze(-1))
                
            weight = F.softmax(torch.cat(weight_list, dim=1), dim=1)

            input_feas = []
            for i in range(n_nodes):
                fea = 0.
                for j in range(hop+1):
                    fea += (weight[i][j]*features_list[j][i]).unsqueeze(0)
                input_feas.append(fea)
            input_feas = torch.cat(input_feas, dim=0)
            input_features = input_features + input_feas
        input_features /= len(r_list)

        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=args.seed)
        y_pred = kmeans.fit_predict(input_features.numpy())
        eva(y, y_pred, hop)


if __name__ == '__main__':
    run(args)
