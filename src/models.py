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


class DNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DNN, self).__init__()

        self.fcn1 = nn.Linear(nfeat, nhid)
        self.fcn2 = nn.Linear(nhid, nclass)
        self.fcn3 = nn.Linear(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.fcn1(x))
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.fcn2(x)
        x = self.fcn3(x)
        
        return F.softmax(x, dim=1)