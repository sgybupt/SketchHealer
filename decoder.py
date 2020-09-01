import torch.nn as nn
import torch
import torch.nn.functional as F
from hyper_params import hp


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # to init **hidden and cell** from z:
        # Linear has no batch_size params, but can compute batch
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(5 + hp.Nz, hp.dec_hidden_size, dropout=hp.dropout)  # 5+128, 512,
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)  # (512, 6 * 20 + 3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            # print(z, z.shape)  # [100, 128], [batch_size, Nz]
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the LSTM with the whole input in one shot
        # and use all outputs contained in 'outputs',
        # while in generate mode we just feed with the last generated sample:
        if self.training:
            # torch.Size([130, 100, 512]) -> torch.Size([13000, 123])
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])  # trajectory  torch.Size([20, 13000, 6])
        params_pen = params[-1]  # pen up/down  torch.Size([13000, 3])
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # preprocess params::
        if self.training:
            len_out = hp.Nmax + 1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)  # torch.Size([130, 100, 20])
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
        q = F.softmax(params_pen).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell
