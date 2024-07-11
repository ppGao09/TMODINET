""" Training and testing of the model
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor,cal_adj_mat_parameter,GraphConstructLoss,normalize_adj
import torch.nn as nn
from dirichlet import DS_Combin,ce_loss

cuda = True if torch.cuda.is_available() else False

# 预处理数据 train test
def prepare_trte_data(data_folder, view_list):

    num_view = len(view_list)

    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)

    # 多组学数据加载
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))    # concatenate用于合并不同的数组

    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))     # 类型转换
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))

    data_train_list = []
    data_test_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))    # torch.cat()把多个tensor拼接

    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_test_list, idx_dict, labels

# 生成邻接矩阵
def gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_adj_mat_tensor(data_te_list[i], adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta,num_class,epoch,train_TCL=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)

    Encoder_list = model_dict["S"](data_list)

    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()

        adj_train = model_dict["GL{:}".format(i + 1)](data_list[i])
        graph_loss = GraphConstructLoss(data_list[i], adj_train, adj_list[i], theta_smooth, theta_degree, theta_sparsity)
        final_adj = neta * adj_train + (1 - neta) * adj_list[i]
        normalized_adj = normalize_adj(final_adj)
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](Encoder_list[i],normalized_adj))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))

        loss_total = ci_loss + graph_loss
        loss_total.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = loss_total.detach().cpu().numpy().item()

    if train_TCL and num_view >= 2:

        Encoder_list = model_dict["S"](data_list)
        optim_dict["D"].zero_grad()
        ci_list = []
        GCN_list = []
        for i in range(num_view):
            adj_train = model_dict["GL{:}".format(i + 1)](data_list[i])
            final_adj = neta * adj_train + (1 - neta) * adj_list[i]
            normalized_adj = normalize_adj(final_adj)
            GCN_list.append(model_dict["E{:}".format(i + 1)](Encoder_list[i],normalized_adj))

        atten_data_list,loss_MOAM= model_dict["TCL"](GCN_list)
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i + 1)](atten_data_list[i]))

        MMlogit, uncertainty, b = model_dict["DS"](ci_list)

        Tem_Coef = epoch * (0.99 / 2500) + 0.01
        loss_CE = torch.mean(criterion(b / Tem_Coef, label))
        loss_un = ce_loss(label, MMlogit, num_class, 0, 1)
        c_loss = torch.mean(loss_un) + loss_CE + 0.8*loss_MOAM

        c_loss.backward()
        optim_dict["D"].step()
        loss_dict["D"] = c_loss.detach().cpu().numpy().item()

    
    return loss_dict
    

def test_epoch(data_list, adj_list, model_dict, neta):

    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)

    Encoder_list = model_dict["S"](data_list)
    ci_list = []
    GCN_list = []
    for i in range(num_view):
        adj_test = model_dict["GL{:}".format(i + 1)](data_list[i])
        final_adj = neta * adj_test + (1 - neta) * adj_list[i]
        normalized_adj = normalize_adj(final_adj)
        GCN_list.append(model_dict["E{:}".format(i + 1)](Encoder_list[i], normalized_adj))

    atten_data_list = model_dict["TCL"](GCN_list,is_test=True)
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i + 1)](atten_data_list[i]))

    MMlogit,uncertainty,b = model_dict["DS"](ci_list)

    return MMlogit, uncertainty.cpu().detach().numpy()

def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, theta_smooth, theta_degree, theta_sparsity, neta, reg):

    test_inverval = 50
    num_view = len(view_list)

    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_gcn_list = [200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 8
        dim_gcn_list = [400,400,200]
    if data_folder == 'KIPAN':
        adj_parameter = 6
        dim_gcn_list = [1000, 1000, 300]
    if data_folder == 'LGG':
        adj_parameter = 6
        dim_gcn_list = [1000, 1000, 400]

    mode = 'weighted-cosine'
    data_tr_list, data_te_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)

    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter)

    dim_list = [x.shape[1] for x in data_tr_list]
    print("\n三组学特征维数")
    print(dim_list)  # [1000, 1000, 503]

    model_dict = init_model_dict(num_view, num_class, dim_list, dim_gcn_list, adj_parameter, mode)

    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c, reg)

    print("\nPretrain GCNs...")
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, theta_smooth, theta_degree,
                    theta_sparsity, neta, num_class, epoch, train_TCL=False)

    print("\nTraining...")
    best_acc = 0
    best_f1_macro = 0
    best_f1_weighted = 0
    best_f1 = 0
    best_auc = 0
    best_uncertainty = 100
    best_result = {"acc": [], "f1": [], "auc": [], "uncertainty": []}
    best_result_1 = {"acc": [], "f1-macro": [], "f1-weighted": [], "uncertainty": []}

    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c, reg)

    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, theta_smooth,
                    theta_degree, theta_sparsity, neta, num_class, epoch)

        if epoch % test_inverval == 0:
            te_prob, uncertainty = test_epoch(data_te_list, adj_te_list, model_dict, neta)

            if num_class == 2:
                if accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)) * 100 >= best_acc:
                    best_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)) * 100
                    best_f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)) * 100
                    best_auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1].detach().numpy()) * 100
                    best_uncertainty = np.mean(uncertainty) * 100
            else:
                if accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)) * 100 >= best_acc:
                    best_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1)) * 100
                    best_f1_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1),
                                             average='macro') * 100
                    best_f1_weighted = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1),
                                                average='weighted') * 100
                    best_uncertainty = np.mean(uncertainty) * 100

    if num_class == 2:

        print("\nResult...")
        print(f"acc:{best_acc}")
        print(f"f1:{best_f1}")
        print(f"auc:{best_auc}")
        print(f"uncertainty:{best_uncertainty}")

        best_result["acc"].append(best_acc)
        best_result["f1"].append(best_f1)
        best_result["auc"].append(best_auc)
        best_result["uncertainty"].append(best_uncertainty)

        print(best_result)


    else:
        print("\nResult...")
        print(f"acc:{best_acc}")
        print(f"f1_macro:{best_f1_macro}")
        print(f"f1_weighted:{best_f1_weighted}")
        print(f"uncertainty:{best_uncertainty}")

        best_result_1["acc"].append(best_acc)
        best_result_1["f1-macro"].append(best_f1_macro)
        best_result_1["f1-weighted"].append(best_f1_weighted)
        best_result_1["uncertainty"].append(best_uncertainty)

        print(best_result_1)
















