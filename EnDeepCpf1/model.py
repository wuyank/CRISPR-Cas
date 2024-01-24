import torch.nn as nn
import torch.nn.functional as F
import torch

class EnCas12a(nn.Module):
    def __init__(self,
                 len_seq,
                 n_filters,
                 n_conv1d,
                 kernel_size,
                 fc_units,
                 n_fc,
                 dropout
                 ) -> None:
        super().__init__()
        self.layer1 = self.conv1d_layer(4, n_filters, kernel_size, 1, int((kernel_size-1)/2))
        self.conv1ds = nn.ModuleList(
            [self.conv1d_layer(n_filters, n_filters, kernel_size, 1, int((kernel_size-1)/2)) for _ in range(n_conv1d - 1)]
        )
        self.layer2 = self.fc_layer(n_filters * len_seq, fc_units, dropout)
        self.fcs = nn.ModuleList(
            [self.fc_layer(fc_units, fc_units, dropout) for _ in range(n_fc - 1)]
        )
        self.out = nn.Sequential(nn.Linear(fc_units, 1), nn.Sigmoid())

    def conv1d_layer(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding), nn.ReLU())
    
    def fc_layer(self, in_units, out_units, dropout):
        return nn.Sequential(nn.Linear(in_units, out_units), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        x = self.layer1(x)
        for conv in self.conv1ds:
            x = conv(x)
        x = x.view(x.size()[0], -1)
        x = self.layer2(x)
        for fc in self.fcs:
            x = fc(x)
        x = self.out(x)
        return x

class EnCas12aCA(nn.Module):
    def __init__(self,
                 len_seq,
                 n_filters,
                 n_conv1d,
                 kernel_size,
                 fc_units,
                 n_fc,
                 dropout,
                 ca_units
                 ) -> None:
        super().__init__()
        self.layer1 = self.conv1d_layer(4, n_filters, kernel_size, 1, int((kernel_size-1)/2))
        self.conv1ds = nn.ModuleList(
            [self.conv1d_layer(n_filters, n_filters, kernel_size, 1, int((kernel_size-1)/2)) for _ in range(n_conv1d - 1)]
        )
        self.layer2 = self.fc_layer(n_filters * len_seq, fc_units, dropout)
        self.fcs = nn.ModuleList(
            [self.fc_layer(fc_units, fc_units, dropout) for _ in range(n_fc - 1)]
        )
        self.ca = nn.Sequential(nn.Linear(1, ca_units), nn.ReLU(), nn.Dropout(dropout))
        self.outca = nn.Sequential(nn.Linear(fc_units + ca_units, 1), nn.Sigmoid())

    def conv1d_layer(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding), nn.ReLU())
    
    def fc_layer(self, in_units, out_units, dropout):
        return nn.Sequential(nn.Linear(in_units, out_units), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, ca):
        x = self.layer1(x)
        for conv in self.conv1ds:
            x = conv(x)
        x = x.view(x.size()[0], -1)
        x = self.layer2(x)
        for fc in self.fcs:
            x = fc(x)
        ca = self.ca(ca)
        out = torch.cat([x, ca], dim=1)
        out = self.outca(out)
        return out

class DeepCpf1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(4, 80, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(80*17, 80), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(80, 40), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(40, 40), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(40, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size()[0], -1)
        x = self.layer2(x)
        return x


class DeepCpf1CA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(4, 80, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(80*17, 80), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(80, 40), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(40, 40), nn.ReLU(),
        )
        self.ca = nn.Sequential(nn.Linear(1, 40), nn.ReLU())
        self.output = nn.Sequential(nn.Dropout(0.3), nn.Linear(80, 1), nn.Sigmoid())

    def forward(self, x, ca):
        x = self.layer1(x)
        x = x.view(x.size()[0], -1)
        x = self.layer2(x)
        ca = self.ca(ca)
        out = torch.cat([x, ca], dim=1)
        out = self.output(out)
        return out

