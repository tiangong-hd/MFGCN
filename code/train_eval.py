from __future__ import division
import time
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam,SGD
from sklearn import metrics
from datasets import *
import pickle as pkl
from sklearn.metrics import roc_curve,auc

def run(dataset, gpu_no, model, epochs, lr, weight_decay, early_stopping, beta, theta, logger=None, save_path=None):
    torch.cuda.set_device(gpu_no)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valacc, val_losses, accs, durations,  auc1,  auc2, trainloss, testloss, avg_tprs, att_history = [], [], [], [], [], [], [], [], [], []
    data = dataset[0]
    
    data = data.to(device)
 
    model.to(device).reset_parameters()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    best_model_state_dict = None
    best_val_loss = float('inf')
    test_acc = 0
    val_loss_history = []
    flag = 0
    
    for epoch in range(1, epochs + 1):
        flag = flag + 1
        X_real = data.x
        X_img = data.x
        train(model, optimizer, data, X_real, X_img, data.edge_index, beta, theta)
        eval_info, logits = evaluate(model, data, X_real, X_img, data.edge_index, weight_decay, device, beta, theta)
        eval_info['epoch'] = epoch

        out, att, x1, com1, com2, x2, emb = model(data, X_real, X_img, edge_index=data.edge_index)

        if logger is not None:
            logger(eval_info)
        train_loss = eval_info['train_loss']
        test_loss = eval_info['test_loss']
        print('train_loss-epoch: {:.4f},test_loss-epoch: {:.4f}'.format(train_loss, test_loss))

        test_tpr1 = eval_info['test_tpr']
        test_fpr1 = eval_info['test_fpr']
        test_auc1 = metrics.auc(test_fpr1, test_tpr1)
        print('test_auc-epoch: {:.4f}'.format(test_auc1))

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            val_acc = eval_info['val_acc']
            test_acc = eval_info['test_acc']
            test_tpr = eval_info['test_tpr']
            test_fpr = eval_info['test_fpr']
            test_auc = metrics.auc(test_fpr, test_tpr)
            best_model_state_dict = model.state_dict()
            att_epoch = att
            x = data.x
        print(best_val_loss, test_acc)
        print(test_auc)
        att_history.append(att_epoch.cpu().detach().numpy())
        val_loss_history.append(eval_info['val_loss'])
        if early_stopping > 0 and epoch > epochs // 2:
            tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
            if eval_info['val_loss'] > tmp.mean().item():
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        avg_fpr = np.linspace(0, 1, 100)
        avg_tprs.append(np.interp(avg_fpr, test_fpr, test_tpr))

        valacc.append(val_acc)
        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        trainloss.append(train_loss)
        testloss.append(test_loss)
    vacc, loss, acc, duration = tensor(valacc), tensor(val_losses), tensor(accs), tensor(durations)
    avg_tpr = np.mean(avg_tprs, axis=0)
    avg_tpr[-1] = 1.0
    avg_tpr[0] = 0.0
    roc_auc = metrics.auc(avg_fpr, avg_tpr)

    print('Val Acc: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.4f} ± {:.4f}, Duration: {:.4f}'.
          format(vacc.mean().item(),
                 loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()
                 ))
    return loss.mean().item(), acc.mean().item(), acc.std().item(), duration.mean().item(), logits, roc_auc, avg_tpr,avg_fpr


def train(model, optimizer, data, X_real, X_img, edge_index, beta, theta):
    model.train()
    optimizer.zero_grad()
    
    out, att, x1, com1, com2, x2, emb = model(data,X_real, X_img, edge_index=edge_index)
    loss_class = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss_com = common_loss(com1, com2)
    loss = loss_class + theta * loss_com
    loss.backward()
    optimizer.step()


def evaluate(model, data, X_real, X_img, edge_index, weight_decay, device, beta, theta):
    model.eval()

    with torch.no_grad():
        logits, att, x1, com1, com2, x2, emb = model(data,X_real, X_img, edge_index=edge_index)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss_class = F.nll_loss(logits[mask], data.y[mask]).item()
        loss_com = common_loss(com1, com2).item()
        loss = loss_class + theta * loss_com
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        y_true = data.y[mask].detach().cpu().numpy()
        score = []
        softlogits = F. log_softmax(logits,dim=1)
        for i in range(len(y_true)):
            s = softlogits[mask][i][1]
            s = s.detach().cpu().numpy()
            score.append(s)
        score = np.array(score)
        fpr, tpr, thresholds = roc_curve(y_true, score)
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
        outs['{}_tpr'.format(key)] = tpr
        outs['{}_fpr'.format(key)] = fpr

    return outs,logits


