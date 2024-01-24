from torch.utils.data import Dataset
import torch


NUCLEOTIDE = {'A':0, 'T':1, 'C':2, 'G':3}

class CasDataset(Dataset):
    def __init__(self, guide_sequences, editting_freqs) -> None:
        super().__init__()
        self.sequences = guide_sequences
        self.y = editting_freqs

        assert len(self.sequences) == len(self.y), f'sequence length ({len(self.sequences)}) dont match editting frequence length ({len(self.y)})'

    def encoding_data(self, sequence):
        '''对序列进行one-hot编码'''
        seq_idx = [NUCLEOTIDE[_] for _ in sequence]
        array = torch.zeros(4, len(sequence))
        array[seq_idx, list(range(len(sequence)))] = 1
        return array
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        return self.encoding_data(self.sequences[index]), self.y[index]

class CasCADataset(Dataset):
    def __init__(self, guide_sequences, editting_freqs, chr_access) -> None:
        super().__init__()
        self.sequences = guide_sequences
        self.chr_access = chr_access
        self.y = editting_freqs

        assert len(self.sequences) == len(self.y), f'sequence length ({len(self.sequences)}) dont match editting frequence length ({len(self.y)})'

    def encoding_data(self, sequence):
        '''对序列进行one-hot编码'''
        seq_idx = [NUCLEOTIDE[_] for _ in sequence]
        array = torch.zeros(4, len(sequence))
        array[seq_idx, list(range(len(sequence)))] = 1
        return array
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        return self.encoding_data(self.sequences[index]), torch.tensor([self.chr_access[index]], dtype=torch.float), self.y[index]

