import torch
import torch.nn as nn
import torchvision
import time
import numpy as np
import random
from hyper_params import hp


class FeatureExtractionBasic(nn.Module):
    def __init__(self):
        super(FeatureExtractionBasic, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 2, 2, 0)  # 64
        self.conv2 = nn.Conv2d(8, 32, 2, 2, 0)  # 32
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 0)  # 16
        self.conv4 = nn.Conv2d(64, 128, 2, 2, 0)  # 8
        self.conv5 = nn.Conv2d(128, 256, 2, 2, 0)  # 4
        self.conv6 = nn.Conv2d(256, 512, 2, 2, 0)  # 2
        self.maxpooling1 = nn.MaxPool2d(2)  # 1
        pass

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv5(x))
        x = nn.ReLU()(self.conv6(x))
        x: torch.Tensor = self.maxpooling1(x)
        x = x.view(-1, 512)
        return x


class FeatureExtraction(nn.Module):
    def __init__(self, graph_num=0, graph_size=0, train=True):
        super().__init__()
        self.graph_num = graph_num
        self.graph_size = graph_size
        assert self.graph_num
        assert self.graph_size
        self.featureGenerator = FeatureExtractionBasic()
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: (batch_size, graph_num, 3, graph_size, graph_size)
        :return:
        """
        if inputs.shape[0] != 1:
            tmp_batch = 1
            tmp_result = []
            inputs = inputs.view(tmp_batch, -1, 1, self.graph_size, self.graph_size)
            for i in range(tmp_batch):
                tmp_result.append(self.featureGenerator(inputs[i]))
            result = torch.cat(tmp_result).view(-1, self.graph_num, 512)  # (batch, 30, 1000)
        else:
            result = self.featureGenerator(inputs.view(-1, 1, self.graph_size, self.graph_size)
                                           ).view(-1, self.graph_num, 512)
        result = self.bn1(result.view(-1, 512)).view(-1, self.graph_num, 512)
        return result


if __name__ == '__main__':
    featureExtractionTest: nn.Module = FeatureExtraction(25, 128).cuda(1)
    fake_img = torch.rand((100, 25, 1, 128, 128)).cuda(1)
    t = time.time()
    for _ in range(2):
        print(_)
        res: torch.Tensor = featureExtractionTest(fake_img)
    print((time.time() - t) / 2)

    print("feature extraction output:", res.shape, res.dtype, res.min(), res.max())
    exit(0)


class GCNProcessor(nn.Module):

    def __init__(self, graph_num, out_f_num, bias_bool=True):
        super().__init__()
        # shapes
        self.graph_num = graph_num
        self.out_f_num = out_f_num
        self.bias_bool = bias_bool
        # params
        self.weight = nn.Parameter(torch.randn(512, self.out_f_num, dtype=torch.float, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(self.graph_num, self.out_f_num, dtype=torch.float, requires_grad=True))
        self.merge = nn.Parameter(torch.randn(1, self.graph_num, dtype=torch.float, requires_grad=True))

        self.bn1 = nn.BatchNorm1d(out_f_num)

        # init params
        # self.params_reset()

    def params_reset(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.merge, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def set_trainable(self, train=True):
        for param in self.parameters():
            param.requires_grad = train

    def forward(self, input_g_f, adj):
        """
        :param input_g_f: (batch, graph_num, in_feature_num)
        :param adj: (batch, graph_num, graph_num)
        :return:
        """
        x = torch.matmul(adj, input_g_f)
        if self.bias_bool:
            x = torch.matmul(x, self.weight) + self.bias
        else:
            x = torch.matmul(x, self.weight)

        # result =  torch.matmul(self.merge, x).squeeze(1)
        result = torch.sum(x, dim=1)
        # print(self.bn1(result)[0])
        return self.bn1(result)


if __name__ == '__main__' and 0:
    graphNum = 3
    i_feature_num = 1000
    o_feature_num = 2
    batch_size = 128
    graph_feature = torch.ones(batch_size, graphNum, i_feature_num).cuda()
    adj_matrix = torch.Tensor(batch_size, graphNum, graphNum).cuda()
    for index in range(batch_size):
        adj_matrix[index] = torch.eye(graphNum)
    gcnTest = GCNProcessor(graphNum, o_feature_num, bias_bool=True)
    gcnTest.cuda()
    res = gcnTest(graph_feature, adj_matrix)
    print(f"gcn result\n", res)
    print(f"gcn processer output:\n", res.shape, res.dtype, res.min(), res.max())


class EncoderGCN(nn.Module):
    def __init__(self, graph_num, graph_size, out_f_num, out_mu_sigma_num,
                 bias_need=False, FE_trainable=False):
        super(EncoderGCN, self).__init__()
        self.graph_num = graph_num
        self.graph_size = graph_size
        self.out_f_num = out_f_num
        self.bias_need = bias_need
        self.out_mu_sigma_num = out_mu_sigma_num
        assert self.graph_num
        assert self.graph_size
        assert self.out_f_num
        assert self.out_mu_sigma_num

        # model
        self.feature_extractor = FeatureExtraction(self.graph_num, self.graph_size, FE_trainable)
        self.gcn = GCNProcessor(self.graph_num, self.out_f_num, self.bias_need)

        # z, mu, sigma
        self.fc_mu = nn.Linear(self.out_f_num, self.out_mu_sigma_num)
        self.fc_sigma = nn.Linear(self.out_f_num, self.out_mu_sigma_num)

    def forward(self, input_imgs, adj_matrix):
        """
        return z, mu, sigma
        :param input_imgs: (batch_size, graph_num, 3, graph_size, graph_size)
        :param adj_matrix: (batch_size, graph_num, graph_num)
        """
        x = self.feature_extractor(input_imgs)
        x = self.gcn(x, adj_matrix)
        final = torch.tanh(x)
        # print(f'final', final, final.shape, final.min(), final.max())

        # generate mu sigma
        mu = self.fc_mu(final)
        sigma = self.fc_sigma(final)
        sigma_e = torch.exp(sigma / 2.)

        # normal sample
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n
        return z, mu, sigma


if __name__ == '__main__':
    batch_size = 5
    fake_img = torch.randn(batch_size, 30, 3, 128, 128)
    fake_img = fake_img / fake_img.max()
    eyes = torch.Tensor(batch_size, 30, 30)
    for i in range(batch_size):
        eyes[i] = torch.eye(30)
    eyes = eyes.cuda()
    fake_img = fake_img.cuda()
    encoderGCN = EncoderGCN(30, 128, 200, 64, True, False).cuda()
    z, mu, sigma = encoderGCN(fake_img, eyes)
    print(f'z', z)
    print(z.shape, z.max(), z.min())
    print(f'mu', mu)
    print(mu.shape, mu.max(), mu.min())
    print(f'sigma', sigma)
    print(sigma.shape, sigma.max(), sigma.min())
