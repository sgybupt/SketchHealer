import torch


class HParams:
    def __init__(self):
        self.data_location = ''#location of  of origin data
        self.category = ["airplane.npz", "angel.npz", "alarm clock.npz", "apple.npz",
                         "butterfly.npz", "belt.npz", "bus.npz",
                         "cake.npz", "cat.npz", "clock.npz", "eye.npz", "fish.npz",
                         "pig.npz", "sheep.npz", "spider.npz", "The Great Wall of China.npz",
                         "umbrella.npz"]
        self.model_save = "model_save"
        
        self.enc_hidden_size = 256  # encoder LSTM h size
        self.dec_hidden_size = 512
        self.Nz = 128  # encoder output size
        self.M = 20  # 采样个数
        self.dropout = 0.0
        self.batch_size = 500
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.99999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.2

        self.max_seq_length = 200  
        self.min_seq_length = 0  

        self.Nmax = 0  # max stroke number of a sketch
        self.graph_number = 1 + 20  # the number of graph for each sketch,first for global
        self.graph_picture_size = 128  # size of graph
        self.out_f_num = 512  # 1000 -> 512
        self.mask_prob = 0.0
        self.use_cuda = torch.cuda.is_available()
        self.res_number = 2


hp = HParams()

