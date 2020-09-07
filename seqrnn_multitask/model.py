# coding: utf-8
# @Author: Huzi Cheng
# @Date: 2017.08.25
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable


# torch.manual_seed(1926)
# H_NOISE_AMP = 0.1

class GRUNetwork(nn.Module):
    """
    GRU network model
    """

    def __init__(self, ninp, nhid, bsz, nlayers, lr, cuda_enabled=False):
        
        super(GRUNetwork, self).__init__()
        
        self.lr      = lr
        self.rnn     = nn.GRU(ninp, nhid, num_layers = nlayers)
        self.ninp    = ninp
        self.nhid    = nhid
        self.nlayers = nlayers
        self.batch_size     = bsz
        self.cuda_enabled = cuda_enabled

        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
                        
        self.decoder   = nn.Linear(nhid, ninp) # output_size == input_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduce=True)
#        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        if self.cuda_enabled:
            torch.cuda.manual_seed_all(int(time.time()))
            self.init_hidden = self.init_hidden_gpu
            self.cuda()
        else:
            torch.manual_seed(int(time.time()))
            self.init_hidden = self.init_hidden_cpu

    def forward(self, input, hidden, bsz = None):
        """
        inputs shape: seq_len, batch, input_size
        outputs shape; seq_len, batch, input_size
        """
        bsz = self.batch_size if bsz == None else bsz
        output, hidden = self.rnn(input, hidden)
        hidden = self.relu(hidden)
        output = self.relu(output)
        output = output.view(bsz, -1) 
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output, hidden 
        
    def init_hidden_cpu(self, bsz = None):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        bsz = self.batch_size if bsz == None else bsz
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_hidden_gpu(self, bsz = None):
        """Init the hidden cells' states before any trial.
        Add perturbation in initial hidden state
        """
        bsz = self.batch_size if bsz == None else bsz
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda())

    
