import pandas as pd
import math
from dataset import CasDataset, CasCADataset
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
import torch


def load_dataset(trainfile, testfile, batch_size):
    traindf = pd.read_csv(trainfile)
    testdf = pd.read_csv(testfile)
    train_seqs = traindf['seq'].tolist()
    train_freqs = traindf['indel_freq'].tolist()
    test_seqs = testdf['seq'].tolist()
    test_freqs = testdf['indel_freq'].tolist()
    train_ds = CasDataset(train_seqs, train_freqs)
    test_ds = CasDataset(test_seqs, test_freqs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

def load_dataset_ca(trainfile, testfile, batch_size):
    traindf = pd.read_csv(trainfile)
    testdf = pd.read_csv(testfile)
    train_seqs = traindf['seq'].tolist()
    train_freqs = traindf['indel_freq'].tolist()
    train_ca = traindf['chromatin_accessibility'].tolist()
    test_seqs = testdf['seq'].tolist()
    test_freqs = testdf['indel_freq'].tolist()
    test_ca = testdf['chromatin_accessibility'].tolist()
    train_ds = CasCADataset(train_seqs, train_freqs, train_ca)
    test_ds = CasCADataset(test_seqs, test_freqs, test_ca)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

def load_dataset_seq34(trainfile, testfile, batch_size):
    traindf = pd.read_csv(trainfile)
    testdf = pd.read_csv(testfile)
    train_seqs = traindf['seq34'].tolist()
    train_freqs = traindf['indel_freq'].tolist()
    test_seqs = testdf['seq34'].tolist()
    test_freqs = testdf['indel_freq'].tolist()
    train_ds = CasDataset(train_seqs, train_freqs)
    test_ds = CasDataset(test_seqs, test_freqs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

def load_dataset_ca_seq34(trainfile, testfile, batch_size):
    traindf = pd.read_csv(trainfile)
    testdf = pd.read_csv(testfile)
    train_seqs = traindf['seq34'].tolist()
    train_freqs = traindf['indel_freq'].tolist()
    train_ca = traindf['chromatin_accessibility'].tolist()
    test_seqs = testdf['seq34'].tolist()
    test_freqs = testdf['indel_freq'].tolist()
    test_ca = testdf['chromatin_accessibility'].tolist()
    train_ds = CasCADataset(train_seqs, train_freqs, train_ca)
    test_ds = CasCADataset(test_seqs, test_freqs, test_ca)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

def train_test(train_loader, test_loader, model, loss_func, optimizer, device, n_epochs, early_stop, model_path, use_ca = False):
    model.to(device)
    loss_func.to(device)
    
    early_stop_count = 0
    best_loss = math.inf
    best_cor = -2.
    best_spearman_cor = -2.
    early_stop_test_y_labels = []
    early_stop_test_ypred_labels = []
    test_pearson_cor_perepoch = []
    test_spearman_cor_perepoch = []
    test_loss_perepoch = []
    train_pearson_cor_perepoch = []
    train_spearman_cor_perepoch = []
    train_loss_perepoch = []
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        y_pred_record = []
        y_record = []
        for items in train_loader:
            optimizer.zero_grad()
            if use_ca:
                x, ca, y = items
                x,ca,y = x.to(device), ca.to(device), y.to(device)
                y = y.float()
                y_pred = model(x, ca)
            else:
                x, y = items
                x,y = x.to(device), y.to(device)
                y = y.float()
                y_pred = model(x)
            loss = loss_func(y_pred.squeeze(1), y)

            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
            y_pred_record.extend(y_pred.flatten().detach().cpu().numpy())
            y_record.extend(y.flatten().detach().cpu())
        mean_train_loss = sum(loss_record) / len(loss_record)
        train_pearson_cor = pearsonr(y_pred_record, y_record)[0]
        train_spearman_cor = spearmanr(y_pred_record, y_record)[0]
        train_pearson_cor_perepoch.append(train_pearson_cor)
        train_spearman_cor_perepoch.append(train_spearman_cor)
        train_loss_perepoch.append(mean_train_loss)

        # validation
        model.eval()
        loss_record = []
        y_pred_record = []
        y_record = []
        for items in test_loader:
            if use_ca:
                x, ca, y = items
                x,ca,y = x.to(device), ca.to(device), y.to(device)
                y = y.float()
                with torch.no_grad():
                    y_pred = model(x, ca)
                    loss = loss_func(y_pred.squeeze(), y)
            else:
                x, y = items
                x,y = x.to(device), y.to(device)
                y = y.float()
                with torch.no_grad():
                    y_pred = model(x)
                    loss = loss_func(y_pred.squeeze(), y)
            loss_record.append(loss.item())
            y_pred_record.extend(y_pred.flatten().detach().cpu().numpy())
            y_record.extend(y.flatten().detach().cpu())
        mean_test_loss = sum(loss_record) / len(loss_record)
        test_pearson_cor = pearsonr(y_pred_record, y_record)[0]
        test_spearman_cor = spearmanr(y_pred_record, y_record)[0]
        test_pearson_cor_perepoch.append(test_pearson_cor)
        test_spearman_cor_perepoch.append(test_spearman_cor)
        test_loss_perepoch.append(mean_test_loss)

        print(f'Epoch {epoch+1}/{n_epochs}:\nTrain loss: {mean_train_loss:.4f} | Train pearson cor: {train_pearson_cor:.4f}  | Train spearman cor: {train_spearman_cor:.4f}')
        print(f'Test  loss: {mean_test_loss:.4f} | Test  pearson cor: {test_pearson_cor:.4f}  | Test  spearman cor: {test_spearman_cor:.4f}\n')

        # early stopping
        if mean_test_loss < best_loss:
        # if test_pearson_cor > best_cor:
        # if test_spearman_cor > best_spearman_cor:
            best_loss = mean_test_loss
            best_cor = test_pearson_cor
            best_spearman_cor = test_spearman_cor
            early_stop_test_y_labels = y_record
            early_stop_test_ypred_labels = y_pred_record
            early_stop_count = 0
            torch.save(model.state_dict(), model_path)

        else:
            early_stop_count += 1
        
        if early_stop_count == early_stop:
            print('****** Model is not improving, stop training. ******\n')
            print(f'Test  loss: {best_loss:.4f} | Test  pearson cor: {best_cor:.4f} | Test  spearman cor: {best_spearman_cor:.4f}\n')
            results = {
                'test_loss':best_loss,
                'test_cor':best_cor,
                'test_spearman_cor':best_spearman_cor,
                'test_loss_perepoch':test_loss_perepoch[:-early_stop],
                'test_pearson_cor_perepoch':test_pearson_cor_perepoch[:-early_stop],
                'test_spearman_cor_perepoch':test_spearman_cor_perepoch[:-early_stop],
                'train_loss_perepoch':train_loss_perepoch[:-early_stop],
                'train_pearson_cor_perepoch':train_pearson_cor_perepoch[:-early_stop],
                'train_spearman_cor_perepoch':train_spearman_cor_perepoch[:-early_stop],
                'early_stop_test_y_labels':early_stop_test_y_labels,
                'early_stop_test_ypred_labels':early_stop_test_ypred_labels
            }
            return results 

def predict(model, test_dl, device, use_ca = False):
    model.to(device)
    model.eval()
    y_pred_record = []
    for items in test_dl:
        if use_ca:
            x, ca, y = items
            x,ca,y = x.to(device), ca.to(device), y.to(device)
            y = y.float()
            with torch.no_grad():
                y_pred = model(x, ca)
        else:
            x, y = items
            x,y = x.to(device), y.to(device)
            y = y.float()
            with torch.no_grad():
                y_pred = model(x)
        y_pred_record.extend(y_pred.flatten().detach().cpu().numpy())
    return y_pred_record

def load_pretrained_model(model, pretrained_pth):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_pth)
    new_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)

    # 只更新最后的全连接层
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fcs.parameters():
        param.requires_grad = True
    for param in model.ca.parameters():
        param.requires_grad = True
    for param in model.outca.parameters():
        param.requires_grad = True

    return model