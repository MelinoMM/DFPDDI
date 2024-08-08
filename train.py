import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve, auc
import os
import random
import matplotlib.pyplot as plt
from losses.Loss_DPL import Loss_DPL
from losses.Loss_IMI import crossview_contrastive_Loss
from losses.normalize import Normalize
from util.plotTSNE import plot_tsne
from util.tsne import visualize_data_tsne_byY_nolabel
from util.tsne import visualize_data_tsne_byX
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# device = torch.device("cuda:1")
# torch.cuda.set_device(device)

set_random_seed(1, deterministic=True)


def train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_dpl = Loss_DPL()
    norm = Normalize(2)
    loss_history = []
    max_auc = 0
    max_f1 = 0
    hsic_x_z_list = []

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_s.to('cuda')
        data_a.to('cuda')

    # Train model
    lbl = data_a.y
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    stoping = 0
    for epoch in range(args.epochs):
        # stoping=0
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (inp) in enumerate(train_loader):

            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            # output, cla_os, cla_os_a, cla_s_a, _ = model(
                # data_o, data_s, data_a, inp)
            output, cla_os, cla_oa, cla_os_a, ori_featrue, final_DDI, final_molecule, cor_edge_fea, cor_fedge_fea= model(
                data_o, data_s, data_a, inp)

            log = torch.squeeze(output)

            loss1 = loss_fct(log, label.long())
            loss_node = b_xent(cla_oa, lbl.float())
            loss_edge = b_xent(cla_os, lbl.float())
            loss_fedge = b_xent(cla_os_a, lbl.float())
            dpl_loss = loss_dpl(norm(log))
            
            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio3 * loss_edge + args.loss_ratio4 * loss_fedge + args.loss_dpl * dpl_loss
            
            loss_history.append(loss_train.cpu().detach().numpy())
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        y_pred_train1 = []
        y_label_train = np.array(y_label_train)
        y_pred_train = np.array(y_pred_train).reshape((-1, 65))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    # print(y_pred_train[i][j])
                    y_pred_train1.append(j)
                    break

        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro', zero_division=1)
        precision1 = precision_score(
            y_label_train, y_pred_train1, average='macro', zero_division=1)

        if not args.fastmode:
            acc_val, f1_val, recall_val, precision_val, loss_val, hsic_x_z = test(
                model, val_loader, data_o, data_s, data_a, args, 0) 
            
            hsic_x_z_list.append(hsic_x_z.cpu().numpy())            
            
            # if acc_val >= max_auc and f1_val>=max_f1
            if acc_val >= max_auc and f1_val >= max_f1:
                model_max = copy.deepcopy(model)
                max_auc = acc_val
                max_f1 = f1_val
                stoping = 0
            else:
                stoping = stoping+1
            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_dpl: {:.4f}'.format(dpl_loss.item()),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(acc),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'recall_val: {:.4f}'.format(recall_val),
                  'precision_val: {:.4f}'.format(precision_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


    filenameXZ = "hsic_x_z" + str(args.loss_ratio2) + "_" + str(args.loss_ratio5) + ".csv"


    count = 1
    while os.path.exists(filenameXZ):
        new_filename = f"{filenameXZ}_{count}.csv"
        count += 1
        filenameXZ = new_filename


    np.savetxt(filenameXZ, hsic_x_z_list, delimiter=",")

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing
    acc_test, f1_test, recall_test, precision_test, loss_test, _ = test(
        model_max, test_loader, data_o, data_s, data_a, args, 1)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test), 'recall_test: {:.4f}'.format(recall_test))


def test(model, loader, data_o, data_s, data_a, args, printfou):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_dpl = Loss_DPL()
    norm = Normalize(2)
    model.eval()
    y_pred = []
    y_label = []
    lbl = data_a.y
    zhongzi = args.zhongzi
    embeddingsX_list = []
    embeddingsZ_list = []

    with torch.no_grad():
        for i, (inp) in enumerate(loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            output, cla_os, cla_oa, cla_os_a, ori_featrue, final_DDI, final_molecule, cor_edge_fea, cor_fedge_fea = model(
                data_o, data_s, data_a, inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.long())
            loss_node = b_xent(cla_oa, lbl.float())
            loss_edge = b_xent(cla_os, lbl.float())
            loss_fedge = b_xent(cla_os_a, lbl.float())
            dpl_loss = loss_dpl(norm(log))
            loss = args.loss_ratio1 * loss1 + args.loss_ratio3 * loss_edge + args.loss_ratio4 * loss_fedge + args.loss_dpl * dpl_loss

                
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            
            # embeddings.append(output.detach().cpu().numpy())
            embeddingsX_list.append(final_molecule)
            embeddingsZ_list.append(final_DDI)

    y_pred_train1 = []
    y_label_train = np.array(y_label)
    y_pred_train = np.array(y_pred).reshape((-1, 65))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro', zero_division=1)
    precision1 = precision_score(
        y_label_train, y_pred_train1, average='macro', zero_division=1)
    y_label_train1 = np.zeros((y_label_train.shape[0], 65))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    y_label_tensor = torch.tensor(y_label_train).cuda()
    embeddingsX = torch.cat(embeddingsX_list, dim=0)
    embeddingsZ = torch.cat(embeddingsZ_list, dim=0)
    hsic_x_z = hsic(embeddingsX, embeddingsZ,sigmaX=2,sigmaY=2)

    auc_hong = 0
    aupr_hong = 0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):

        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:
            auc_hong = auc_hong + \
                roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape(
        (-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(
        y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)
    
    if printfou == 1:
        with open(args.out_file, 'a') as f:
            f.write(f"{args.loss_ratio2},{args.loss_ratio3},{args.loss_ratio4},{args.loss_ratio5},{args.beta},{args.sigma},{acc},{f1_score1},{recall1},{precision1},{auc1},{aupr},{auc_macro},{aupr_macro}\n")
    return acc, f1_score1, recall1, precision1, loss, hsic_x_z
