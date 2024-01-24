from model import EnCas12a, EnCas12aCA
from functions import load_dataset, load_dataset_ca, train_test
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
import pandas as pd
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',
                        required=True,
                        help='train datas which must contain "seq" and "indel_freq" two columns')
    parser.add_argument('--val_file',
                        required=True,
                        help='validate datas which must contain "seq" and "indel_freq" two columns')
    parser.add_argument('--output_file',
                        required=True,
                        help='predicted results of validata datas')
    parser.add_argument('--model_pth',
                        required=True,
                        help='trained model weights')
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
    model = EnCas12a(**model_args)
    train_dl, test_dl = load_dataset(args.train_file, args.val_file, args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    results = train_test(
            train_dl,
            test_dl,
            model,
            loss_func,
            optimizer,
            device,
            args.n_epochs,
            args.early_stop,
            args.model_pth,
            use_ca=False
        )
    valdf = pd.read_csv(args.val_file)
    valdf['y_pred'] = results['early_stop_test_ypred_labels']
    valdf.to_csv(args.output_file, index=False)




if __name__ == '__main__':
    main()

