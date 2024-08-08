import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv
import numpy as np
import csv
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1, deterministic=True)

def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        """
        This function is used to estimate the difference between two graphs. In 
        this function, we compute the similarity of a positive graph h_pl and 
        context infomation c, denoted sc_1. Similarly, sc_2 means the simlarity 
        between negative graph and c. Finally, we will get a [z,2] shape matrix
        by concatenate sc_1 and sc_2, called logits.
        c: context infomation
        h_pl: positive sample (graph)
        h_mi: negative sample (graph)
        """
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class dfpddi(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout, zhongzi):
        super(dfpddi, self).__init__()

        self.encoder_o1 = RGCNConv(feature, hidden1, num_relations=65)
        self.encoder_o2 = RGCNConv(hidden1, hidden2, num_relations=65)

        self.attt = torch.zeros(2)
        self.attt[0] = 0.5
        self.attt[1] = 0.5
        self.attt = nn.Parameter(self.attt)
        self.disc = Discriminator(hidden2 * 2)

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.mlp = nn.ModuleList([nn.Linear(192, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 65)
                                  ])

        drug_list = []
        with open('data/drug_listxiao.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                drug_list.append(row[0])
        features = np.load('trimnet/drug_emb_trimnet' + str(zhongzi) + '.npy')

        ids = np.load('trimnet/drug_idsxiao.npy')
        ids = ids.tolist()
        features1 = []
        for i in range(len(drug_list)):
            features1.append(features[ids.index(drug_list[i])])
        features1 = np.array(features1)

        # 572*128, drug features sorted by drug index
        self.features1 = torch.from_numpy(features1).cuda()

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)

        return vectors

    def forward(self, data_o, data_s, data_a, idx):

        # RGCN for DDI event graph and two corrupted graph
        x_o, adj, e_type = data_o.x, data_o.edge_index, data_o.edge_type
        e_type1 = data_a.edge_type
        e_type = torch.tensor(e_type, dtype=torch.int64)
        e_type1 = torch.tensor(e_type1, dtype=torch.int64)
        adj2 = data_s.edge_index
        x_a = data_s.x

        # x1_o and x2_o
        x1_o = F.relu(self.encoder_o1(x_o, adj, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_os = x1_o
        x2_o = self.encoder_o2(x1_os, adj, e_type)
        x2_os = x2_o
        
        # corrupt node features
        x1_o_a = F.relu(self.encoder_o1(x_a, adj, e_type))
        x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
        x1_os_a = x1_o_a
        x2_o_a = self.encoder_o2(x1_os_a, adj, e_type)
        x2_os_a = x2_o_a

        # corrupt edges 
        x1_o_a_a = F.relu(self.encoder_o1(x_o, adj, e_type1))
        x1_o_a_a = F.dropout(x1_o_a_a, self.dropout, training=self.training)
        x1_os_a_a = x1_o_a_a
        x2_o_a_a = self.encoder_o2(x1_os_a_a, adj, e_type1)
        x2_os_a_a = x2_o_a_a

        # corrupt nodes and edges
        x1_a_a = F.relu(self.encoder_o1(x_a, adj, e_type1))
        x1_a_a = F.dropout(x1_a_a, self.dropout, training=self.training)
        x1_s_a_a = x1_a_a
        x2_a_a = self.encoder_o2(x1_s_a_a, adj, e_type1)
        x2_s_a_a = x2_a_a

        # readout
        h_os = self.read(x2_os)
        h_os = self.sigm(h_os)
        
        # contrastive learning: node-edge, node-mix
        ret_os = self.disc(h_os, x2_os, x2_os_a_a) # edges
        ret_oa = self.disc(h_os, x2_os, x2_os_a) # nodes
        ret_os_a = self.disc(h_os, x2_os, x2_s_a_a) # edge & nodes

        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]

        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        # layer attnetion
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1) # [572, 96]

        entity1 = final[aa] # entity1 is the drug v final feature learned from DDI event graph, [256, 96]
        entity2 = final[bb]
        final_DDI = torch.cat((entity1, entity2), dim=1)  # final feature learned from DDI graph, [256, 192]
        
        # skip connection
        entity1_res = self.features1[aa].to('cuda')  # entity1_res is the drug v's feature learned from drug molecule graph, [256, 128]
        entity2_res = self.features1[bb].to('cuda')
        
        final_molecule = torch.cat((entity1_res, entity2_res), dim=1) # final feature learned from molecule graph, [256, 192]

        feature = self.MLP(final_DDI, 7) # [256, 65]
        log = feature

        return log, ret_os, ret_oa, ret_os_a, x2_os, final_DDI, final_molecule, x2_os_a_a, x2_s_a_a
