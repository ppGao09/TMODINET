""" Componets of the model
"""
import csv
import math
import os
import sys

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import LinearLayer
from utils import gen_adj_mat_tensor, cal_adj_mat_parameter
from contrastive_loss import TripleContrastiveLoss
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import MultiheadAttention


def KL(alpha, c):
    """Compute the KL divergence between Dirichlet distribution and a uniform Dirichlet distribution.

    Args:
        alpha (Tensor): Parameter of the Dirichlet distribution.
        c (int): Number of classes or categories.

    Returns:
        Tensor: Computed KL divergence.
    """
    beta = torch.ones((1, c))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    """Compute a modified cross-entropy loss incorporating Dirichlet distribution characteristics.

    Args:
        p (Tensor): Predicted probabilities.
        alpha (Tensor): Parameter of the Dirichlet distribution.
        c (int): Number of classes or categories.
        global_step (int): Current global step during training.
        annealing_step (int): Step at which annealing effect is maximized.

    Returns:
        Tensor: Computed modified cross-entropy loss.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.views = len(in_dim)
        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, data_list):

        FeatureInfo, feature= dict(), dict()
        for view in range(self.views):
            data_list[view] = data_list[view].squeeze(0)
            FeatureInfo[view] = torch.sigmoid(self.dropout(
                self.FeatureInforEncoder[view](data_list[view])))
            feature[view] = data_list[view] * FeatureInfo[view]

        return feature

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)
        self.fc = nn.Softplus()

    def forward(self, x):
        x = self.clf(x)
        h = self.fc(x)

        return h

class DS_Un(nn.Module):
    def __init__(self, num_view, num_cls):
        super().__init__()
        self.num_cls = num_cls
        self.num_view = num_view
        self.clf = nn.Sequential(nn.Linear(3, 3))
        self.clf.apply(xavier_init)
    def DS_Combin(self, alpha, classes):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a, u_a, b_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a, u_a,b_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u_a,b_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a, u_a,b_a

    def late_fusion(self, evidence):
        # 动态融合
        out,energy,conf = dict(),dict(),dict()
        out = evidence
        for v_num in range(len(evidence)):
            energy[v_num] = torch.log(torch.sum(torch.exp(out[v_num]), dim=1))
            conf[v_num] = torch.reshape(energy[v_num] / 10, (-1, 1))  # reshape(-1,1)代表将数组重整为一个一列的二维数组

        return conf

    def forward(self, c_list):
        alpha= dict()
        conf = self.late_fusion(c_list)
        for v_num in range(len(c_list)):
            c_list[v_num] = conf[v_num] * c_list[v_num]
            alpha[v_num] = c_list[v_num] + 1

        alpha_a,u_a,b_a=self.DS_Combin(alpha,self.num_cls)

        return alpha_a,u_a,b_a

class GraphLearn(nn.Module):
    def __init__(self, input_dim, adj_parameter, mode):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Sequential(nn.Linear(input_dim, 1))
        self.p = nn.Sequential(nn.Linear(input_dim, input_dim))

        self.w.apply(xavier_init)
        self.p.apply(xavier_init)

        self.adj_metric = "cosine"  # cosine distance
        self.adj_parameter = adj_parameter


    def forward(self,x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)

        if self.mode == 'adaptive-learning':
            x = x.repeat_interleave(num, dim=0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim=1)

        elif self.mode == 'weighted-cosine':
            x = self.p(x)
            x_norm = F.normalize(x, dim=-1)
            adj_parameter_adaptive = cal_adj_mat_parameter(self.adj_parameter, x_norm, self.adj_metric)
            output = gen_adj_mat_tensor(x_norm, adj_parameter_adaptive, self.adj_metric)

        return output



class Contrastive_Learning_mechanism(nn.Module):
    def __init__(self):
        super().__init__()

        self.Triple_Contrastive_loss = TripleContrastiveLoss()

    def forward(self,input_list,is_test=False):

        if not is_test:
            #################对比学习 方法二####################
            MMLoss = self.Triple_Contrastive_loss(input_list)
            ####################################################
            # MMLoss = 0
            return input_list,MMLoss

        return input_list

def init_model_dict(num_view, num_class, dim_list, dim_he_list, adj_parameter, mode, gcn_dopout=0.5):
    model_dict = {}
    model_dict["S"] = Encoder(dim_list, gcn_dopout)
    for i in range(num_view):
        model_dict['GL{:}'.format(i+1)] = GraphLearn(dim_list[i], adj_parameter, mode)
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)

    if num_view >= 2:
        model_dict["TCL"] = Contrastive_Learning_mechanism()
        model_dict["DS"] = DS_Un(num_view, num_class)
    return model_dict


def init_optim(num_view, model_dict, lr_e, lr_c,reg):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(list(model_dict['GL{:}'.format(i+1)].parameters()) +
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e, weight_decay=reg)
    if num_view >= 2:
        optim_dict["D"] = torch.optim.Adam(list(model_dict["TCL"].parameters()) + list(model_dict["DS"].parameters()), lr=lr_c, weight_decay=reg)

    return optim_dict