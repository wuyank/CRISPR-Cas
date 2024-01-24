from model import EnCas12a, EnCas12aCA
from dataset import CasDataset, CasCADataset
from functions import predict
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        required=True,
                        help='datas for predicting')
    parser.add_argument('--output_file',
                        required=True,
                        help='predicted results')
    parser.add_argument('--model_pth',
                        required=True,
                        help='trained model weights')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size')
    parser.add_argument('--use_ca',
                        action='store_true',
                        help='using chromatin accessibility information')
    return parser.parse_args()


def main():
    args = parse_args()
    indf = pd.read_csv(args.input_file)
    seqs = indf['seq'].tolist()
    freqs = [0] * indf.shape[0]
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
    if args.use_ca:
        model_args['ca_units'] = ca_units
        model = EnCas12aCA(**model_args)
        model.load_state_dict(torch.load(args.model_pth, map_location='cpu'))
        ca = indf['chromatin_accessibility'].tolist()
        ds = CasCADataset(seqs, freqs, ca)
    else:
        model = EnCas12a(**model_args)
        model.load_state_dict(torch.load(args.model_pth, map_location='cpu'))
        ds = CasDataset(seqs, freqs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    y_pred = predict(model, dl, 'cpu', use_ca=args.use_ca)
    indf['y_pred'] = y_pred
    indf.to_csv(args.output_file, index=False)
    



if __name__ == '__main__':
    main()