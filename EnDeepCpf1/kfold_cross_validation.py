from model import EnCas12a, EnCas12aCA
from dataset import CasDataset, CasCADataset
from functions import load_dataset, load_dataset_ca, train_test
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import numpy as np
import math


def kfold_split(datas, labels, kfold, rand_seed=None):
    ndata = len(datas)
    assert ndata == len(labels), f'Number of datas ({ndata}) dont match number of labels ({len(labels)}).'
    idxs = np.array(range(ndata))
    if rand_seed:
        np.random.seed(rand_seed)
    np.random.shuffle(idxs)
    fold_size = math.ceil(ndata / kfold)
    for i in range(kfold):
        if (i + 1) * fold_size > ndata:
            train_idx = idxs[: i * fold_size]
            test_idx = idxs[i * fold_size:]  
        else:
            train_idx = idxs[list(range(i*fold_size)) + list(range((i+1) * fold_size, ndata))]
            test_idx = idxs[i * fold_size : (i+1) * fold_size]
        yield [datas[_] for _ in train_idx], [labels[_] for _ in train_idx], [datas[_] for _ in test_idx], [labels[_] for _ in test_idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',
                        required=True,
                        help='train datas which must contain "seq" and "indel_freq" two columns')
    parser.add_argument('--output_file',
                        required=True,
                        help='output csv file of cross validation')
    parser.add_argument('--k_fold',
                        required=True,
                        type=int,
                        help='fold numbers for cross validation')
    parser.add_argument('--model_dir',
                        required=True,
                        help='dir to save trained model weights')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=1000,
                        help='max training epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='learning rate')
    parser.add_argument('--early_stop',
                        type=int,
                        default=10,
                        help='early stop patience for model no longer getting better')
    parser.add_argument('--random_seed',
                        type=int,
                        default=None,
                        help='random seed for k-fold split')
    return parser.parse_args()

def main():
    args = parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    loss_func = nn.MSELoss(reduction='mean')
    ### model param ###
    len_seq = 31
    n_filters = 128
    kernel_size = 5
    n_conv1d = 2
    dropout = 0.3
    fc_units = 128
    n_fc = 2
    ca_units = 64
    ###
    model_args = {
            'n_filters':n_filters, 'kernel_size':kernel_size, 'n_conv1d':n_conv1d, 
            'len_seq':len_seq, 'dropout':dropout, 'fc_units':fc_units, 
            'n_fc':n_fc
        }
    
    data_file = args.train_file
    output_file = args.output_file
    k_fold = args.k_fold
    model_dir = args.model_dir
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.lr
    early_stop = args.early_stop
    random_seed = args.random_seed

    datadf = pd.read_csv(data_file)
    seqs = datadf['seq'].tolist()
    freqs = datadf['indel_freq'].tolist()

    kfold_idx = 0
    idx_list = []
    test_loss_list = []
    test_pr_list = []
    test_sr_list = []
    test_loss_perepoch_list = []
    test_pr_perepoch_list = []
    test_sr_perepoch_list = []
    train_loss_perepoch_list = []
    train_pr_perepoch_list = []
    train_sr_perepoch_list = []

    for train_seqs, train_freqs, test_seqs, test_freqs in kfold_split(seqs,
                                                                      freqs,
                                                                      k_fold,
                                                                      random_seed):
        model = EnCas12a(**model_args)
        train_ds = CasDataset(train_seqs, train_freqs)
        test_ds = CasDataset(test_seqs, test_freqs)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        model_pth = f'{model_dir}/k_fold_{kfold_idx}.pth'
        results = train_test(
            train_dl,
            test_dl,
            model,
            loss_func,
            optimizer,
            device,
            n_epochs,
            early_stop,
            model_pth,
            use_ca=False
        )
        idx_list.append(kfold_idx)
        test_loss_list.append(results['test_loss'])
        test_pr_list.append(results['test_cor'])
        test_sr_list.append(results['test_spearman_cor'])
        test_loss_perepoch_list.append('_'.join([f'{i:4f}' for i in results['test_loss_perepoch']]))
        test_pr_perepoch_list.append('_'.join([f'{i:4f}' for i in results['test_pearson_cor_perepoch']]))
        test_sr_perepoch_list.append('_'.join([f'{i:4f}' for i in results['test_spearman_cor_perepoch']]))
        train_loss_perepoch_list.append('_'.join([f'{i:4f}' for i in results['train_loss_perepoch']]))
        train_pr_perepoch_list.append('_'.join([f'{i:4f}' for i in results['train_pearson_cor_perepoch']]))
        train_sr_perepoch_list.append('_'.join([f'{i:4f}' for i in results['train_spearman_cor_perepoch']]))
        kfold_idx += 1
    outdf = pd.DataFrame(
        {
            'kfold_idx':idx_list,
            'test_loss':test_loss_list,
            'test_pearson_cor':test_pr_list,
            'test_spearman_cor':test_sr_list,
            'test_loss_perepoch':test_loss_perepoch_list,
            'test_pearson_cor_perepoch':test_pr_perepoch_list,
            'test_spearman_cor_perepoch':test_sr_perepoch_list,
            'train_loss_perepoch':train_loss_perepoch_list,
            'train_pearson_cor_perepoch':train_pr_perepoch_list,
            'train_spearman_cor_perepoch':train_sr_perepoch_list
        }
    )
    outdf.to_csv(output_file, index=0)


if __name__ == '__main__':
    main()