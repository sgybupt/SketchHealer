import os

from hyper_params import hp
import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
from encoder import EncoderGCN
from decoder import DecoderRNN
from utils.sketch_processing import make_graph


################################# load and prepare data
class SketchesDataset:
    def __init__(self, path: str, category: list, mode="train"):
        self.sketches = None
        self.sketches_normed = None
        self.max_sketches_len = 0
        self.path = path
        self.category = category
        self.mode = mode

        tmp_sketches = []
        for c in self.category:
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            print(f"dataset: {c} added.")
        data_sketches = np.concatenate(tmp_sketches)
        print(f"length of trainset: {len(data_sketches)}")

        data_sketches = self.purify(data_sketches)  # data clean.  # remove toolong and too stort sketches.
        self.sketches = data_sketches.copy()
        self.sketches_normed = self.normalize(data_sketches)
        self.Nmax = self.max_size(data_sketches)  # max size of a sketch.

    def max_size(self, sketches):
        """返回所有sketch中 转折最多的一个sketch"""
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches):
        data = []
        for sketch in sketches:
            if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
        return data

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(sketches)
        for sketch in sketches:
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data

    def make_batch(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        batch_idx = np.random.choice(len(self.sketches_normed), batch_size)
        batch_sketches = [self.sketches_normed[idx] for idx in batch_idx]
        batch_sketches_graphs = [self.sketches[idx] for idx in batch_idx]
        sketches = []
        lengths = []
        graphs = []  # (batch_size * graphs_num_constant, x, y)
        adjs = []
        index = 0
        for _sketch in batch_sketches:
            len_seq = len(_sketch[:, 0])  # sketch
            new_sketch = np.zeros((self.Nmax, 5))  # new a _sketch, all length of sketch in size is Nmax.
            new_sketch[:len_seq, :2] = _sketch[:, :2]

            # set p into one-hot.
            new_sketch[:len_seq - 1, 2] = 1 - _sketch[:-1, 2]
            new_sketch[:len_seq, 3] = _sketch[:, 2]

            # len to Nmax set as 0,0,0,0,1
            new_sketch[(len_seq - 1):, 4] = 1
            new_sketch[len_seq - 1, 2:4] = 0  # x, y, 0, 0, 1
            lengths.append(len(_sketch[:, 0]))  # lengths is _sketch length, not new_sketch length.
            sketches.append(new_sketch)
            index += 1

        for _each_sketch in batch_sketches_graphs:
            _graph_tensor, _adj_matrix = make_graph(_each_sketch, graph_num=hp.graph_number,
                                                    graph_picture_size=hp.graph_picture_size, mask_prob=hp.mask_prob)
            graphs.append(_graph_tensor)
            adjs.append(_adj_matrix)

        if hp.use_cuda:
            batch = torch.from_numpy(np.stack(sketches, 1)).cuda().float()  # (Nmax, batch_size, 5)
            graphs = torch.from_numpy(np.stack(graphs, 0)).cuda().float()  # (batch_size, len, 5)
            adjs = torch.from_numpy(np.stack(adjs, 0)).cuda().float()

        else:
            batch = torch.from_numpy(np.stack(sketches, 1)).float()  # (Nmax, batch_size, 5)
            graphs = torch.from_numpy(np.stack(graphs, 0)).float()
            adjs = torch.from_numpy(np.stack(adjs, 0)).float()

        return batch, lengths, graphs, adjs


sketch_dataset = SketchesDataset(hp.data_location, hp.category, "train")
hp.Nmax = sketch_dataset.Nmax


def sample_bivariate_normal(mu_x: torch.Tensor, mu_y: torch.Tensor,
                            sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                            rho_xy: torch.Tensor, greedy=False):
    mu_x = mu_x.item()
    mu_y = mu_y.item()
    sigma_x = sigma_x.item()
    sigma_y = sigma_y.item()
    rho_xy = rho_xy.item()
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]

    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)

    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, epoch, name='_output_'):
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    name = f"./model_save/" + str(epoch) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


################################# encoder and decoder modules


class Model:
    def __init__(self):
        if hp.use_cuda:
            self.encoder: nn.Module = EncoderGCN(hp.graph_number, hp.graph_picture_size, hp.out_f_num, hp.Nz,
                                                 bias_need=False).cuda()
            self.decoder: nn.Module = DecoderRNN().cuda()
        else:
            self.encoder: nn.Module = EncoderGCN(hp.graph_number, hp.graph_picture_size, hp.out_f_num, hp.Nz,
                                                 bias_need=False)
            self.decoder: nn.Module = DecoderRNN()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def lr_decay(self, optimizer: optim):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > hp.min_lr:
                param_group['lr'] *= hp.lr_decay
        return optimizer

    def make_target(self, batch, lengths):
        """
        batch torch.Size([129, 100, 5])  Nmax batch_size
        """
        if hp.use_cuda:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).cuda().unsqueeze(
                0)  # torch.Size([1, 100, 5])
        else:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).unsqueeze(0)  # max of len(strokes)

        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(hp.Nmax + 1, batch.size()[1])
        for indice, length in enumerate(lengths):  # len(lengths) = batchsize
            mask[:length, indice] = 1
        if hp.use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:, :, 0]] * hp.M, 2)  # torch.Size([130, 100, 20])
        dy = torch.stack([batch.data[:, :, 1]] * hp.M, 2)  # torch.Size([130, 100, 20])
        p1 = batch.data[:, :, 2]  # torch.Size([130, 100])
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)  # torch.Size([130, 100, 3])
        return mask, dx, dy, p

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths, graphs, adjs = sketch_dataset.make_batch(hp.batch_size)
        # print(batch, lengths)

        # encode:
        # z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)  # in here, Z is sampled from N(mu, sigma)
        z, self.mu, self.sigma = self.encoder(graphs, adjs)  # in here, Z is sampled from N(mu, sigma)
        # torch.Size([100, 128]) torch.Size([100, 128]) torch.Size([100, 128])
        # print(z.shape, self.mu.shape, self.sigma.shape)

        # create start of sequence:
        if hp.use_cuda:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).cuda().unsqueeze(0)
            # torch.Size([1, 100, 5])
        else:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch], 0)  # torch.Size([130, 100, 5])
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z] * (hp.Nmax + 1))  # torch.Size([130, 100, 128])
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack], 2)  # torch.Size([130, 100, 133])

        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _ = self.decoder(inputs, z)

        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1 - (1 - hp.eta_min) * (hp.R ** epoch)  # self.eta_step = 1 - (1 - hp.eta_min) * hp.R
        # compute losses:
        # LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p, epoch)
        # loss = LR + LKL
        loss = LR
        # gradient step
        loss.backward()  # all torch.Tensor has backward.
        # gradient cliping
        nn.utils.clip_grad_norm(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(), hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch % 1 == 0:
            # print('epoch', epoch, 'loss', loss.item(), 'LR', LR.item(), 'LKL', LKL.item())
            print('gcn, epoch -> ', epoch, 'loss', loss.item(), 'LR', LR.item())
            self.encoder_optimizer = self.lr_decay(self.encoder_optimizer)  # modify optimizer after one step.
            self.decoder_optimizer = self.lr_decay(self.decoder_optimizer)
        if epoch == 0:
            return
        if epoch % 500 == 0:
            self.conditional_generation(epoch)
        if epoch % 1000 == 0:
            self.save(epoch)

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy ** 2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)  # torch.Size([130, 100, 20])
        # stroke
        LS = -torch.sum(mask * torch.log(1e-3 + torch.sum(self.pi * pdf, 2))) / float((hp.Nmax + 1) * hp.batch_size)
        # position
        LP = -torch.sum(p * torch.log(1e-3 + self.q)) / float((hp.Nmax + 1) * hp.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma)) \
              / float(hp.Nz * hp.batch_size)
        if hp.use_cuda:
            KL_min = torch.Tensor([hp.KL_min]).cuda().detach()
        else:
            KL_min = torch.Tensor([hp.KL_min]).detach()
        return hp.wKL * self.eta_step * torch.max(LKL, KL_min)

    def save(self, epoch):
        # sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
                   f'./model_save/encoderRNN_epoch_{epoch}.pth')
        torch.save(self.decoder.state_dict(), \
                   f'./model_save/decoderRNN_epoch_{epoch}.pth')

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self, epoch):
        batch, lengths, graphs, adjs = sketch_dataset.make_batch(1)
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        z, _, _ = self.encoder(graphs, adjs)
        if hp.use_cuda:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
        else:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(hp.Nmax):
            input = torch.cat([s, z.unsqueeze(0)], 2)  # start of stroke concatenate with z
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, hidden, cell = \
                self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            # ------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print(i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence, epoch)

    def sample_next_state(self):
        """
        softmax
        """

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(1e-3 + pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)  # get samples.
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1
        if hp.use_cuda:
            return next_state.cuda().view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
        else:
            return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2


if __name__ == "__main__":
    model = Model()
    epoch_load = 0
    for epoch in range(500001):
        if epoch <= epoch_load:
            continue
        if epoch_load:
            model.load(f'./model_save/encoderRNN_epoch_{epoch_load}.pth',
                       f'./model_save/decoderRNN_epoch_{epoch_load}.pth')
        model.train(epoch)

    '''
    model.load('encoder.pth','decoder.pth')
    model.conditional_generation(0)
    '''
