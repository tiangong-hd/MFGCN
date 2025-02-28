#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch.nn.functional as F
from torch import Tensor
from datasets import *
from train_eval import run
import pickle as pkl
import shutil
from torch_geometric.nn import GCNConv
from layers import GraphConvolution
from complex_relu import complex_relu_layer
from MagNetConv import MagNetConv
from typing import Optional, Tuple, Any

import torch.nn as nn

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DawnNet')
parser.add_argument('--gpu-no', type=int, default=0)
parser.add_argument('--epochs', type=int, default=500)
# 学习率 DawnNet0.013 kegg0.03
parser.add_argument('--lr', type=float, default=0.015)
parser.add_argument('--weight_decay', type=float, default=0.00008)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--theta', type=float, default=0.001)
parser.add_argument('--recache', action="store_true", help="clean up the old adj data", default=True)
parser.add_argument('--normalize-features', action="store_true", default=False)
parser.add_argument('--adj-type', type=str, default='or')
parser.add_argument('--cv-runs', help='Number of cross validation runs', type=int, default=5)
parser.add_argument('--hidden1', type=int, default=32)
parser.add_argument('--hidden2', type=int, default=16)
parser.add_argument('--feat', type=int, default=64)

args = parser.parse_args()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 32)
        self.gc3 = GraphConvolution(32, 16)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.gc3.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class MagNet_model(nn.Module):

    def __init__(self, dataset, hidden: int = 8, q: float = 0.25, K: int = 3, label_dim: int = 2,
                 activation: bool = False, trainable_q: bool = False, layer: int = 2, dropout: float = 0.6,
                 normalization: str = 'sym', cached: bool = False):
        super(MagNet_model, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MagNetConv(in_channels=dataset.num_features, out_channels=32, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        chebs.append(MagNetConv(in_channels=32, out_channels=16, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))
        chebs.append(MagNetConv(in_channels=16, out_channels=hidden, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))

        self.Chebs = chebs
        self.SGCN = GCN(args.feat, args.hidden1, args.hidden2, args.dropout)
        self.CGCN = GCN(args.feat, args.hidden1, args.hidden2, args.dropout)
        self.a = nn.Parameter(torch.zeros(size=(args.hidden2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(args.hidden2)
        self.tanh = nn.Tanh()
        self.MLP = nn.Sequential(
            nn.Linear(args.hidden2, 16),
            nn.Linear(16, 8),
            nn.Linear(8, dataset.num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.SGCN.reset_parameters()
        self.CGCN.reset_parameters()

    def forward(self, data, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: Optional[torch.LongTensor] = None) -> Tuple[Any, Any, Tensor, Any, Any, Any, Any]:

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        fadj, sadj = load_graph(data)
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x1 = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.SGCN(x, fadj)
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)
        Xcom = (com1 + com2) / 2
        emb = torch.stack([x1, x2, Xcom], dim=1)
        emb, att = self.attention(emb)
        x = self.MLP(emb)
        return x, att, x1, com1, com2, x2, emb

def run_IDGCN(dataset, gpu_no, save_folder="saved_models"):
    cv_loss, cv_acc, cv_std, cv_time, output, data, tprs = [], [], [], [], [], [], []
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for cv_run in range(args.cv_runs):
        print('cross validation for the {}th run'.format(cv_run + 1))
        citation_dataset = get_citation_dataset(dataset, args.alpha, args.recache, args.normalize_features,
                                                cv_run, args.adj_type)
        val_loss, test_acc, test_std, time, logits, roc_auc, avg_tpr, avg_fpr = run(citation_dataset, gpu_no,
                                                                                    MagNet_model(citation_dataset),
                                                                                    args.epochs, args.lr,
                                                                                    args.weight_decay,
                                                                                    args.early_stopping, args.beta,
                                                                                    args.theta)

        cv_loss.append(val_loss)
        cv_acc.append(test_acc)
        cv_std.append(test_std)
        cv_time.append(time)
        output.append(logits)
        mean_fpr = np.linspace(0, 1, 100)
        tprs.append(np.interp(mean_fpr, avg_fpr, avg_tpr))
        plt.plot(avg_fpr, avg_tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.3f)' % (cv_run + 1, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

    cv_loss = np.mean(cv_loss)
    cv_acc = np.mean(cv_acc)
    cv_std = np.mean(cv_std)
    cv_time = np.mean(cv_time)

    return cv_loss, cv_acc, cv_std, cv_time, output, data, mean_fpr, mean_tpr, mean_auc


if __name__ == '__main__':
    dataset_name = ["DawnNet"]
    # DawnNet  kegg  RegNetwork(数据集大，本地GPU空间不足)
    outputs = ['val_loss', 'test_acc', 'test_std', 'time']
    result = pd.DataFrame(np.arange(len(outputs) * len(dataset_name), dtype=np.float32).reshape(
        (len(dataset_name), len(outputs))), index=dataset_name, columns=outputs)
    for dataset in dataset_name:
        loss, acc, std, time, logits, data, mean_fpr, mean_tpr, mean_auc = run_IDGCN(dataset, args.gpu_no)
        result.loc[dataset]['loss_mean'] = loss
        result.loc[dataset]['acc_mean'] = acc
        print('ACC: {:.4f}'.format(acc))
        result.loc[dataset]['std_mean'] = std
        result.loc[dataset]['time_mean'] = time

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'result')
        if osp.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        output_path = osp.join(path, '{}.pkl'.format(args.dataset))
        with open(output_path, 'wb') as f:
            pkl.dump([logits, acc], f)

# nvidia-smi
# nvidia-smi -q

# htop
