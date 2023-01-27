import os
import math
import random
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
from torch import Tensor
from torch.autograd import Variable
from typing import Tuple

cuda = True if torch.cuda.is_available() else False
torch.manual_seed(2022)
if cuda:
    torch.cuda.manual_seed_all(2022)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class RLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, state_table):
        super(RLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        self.w_ic = Parameter(Tensor(hidden_size, input_size))
        self.w_hc = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ic = Parameter(Tensor(hidden_size, 1))
        self.b_hc = Parameter(Tensor(hidden_size, 1))

        self.fc = nn.Linear(hidden_size, output_size, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.state_tatble = state_table
        self.MTU = 1500

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform(weight, -stdv, stdv)

    def forward(self, inputs: Tensor, state: Tuple[Tensor] = None, train: bool = True):
        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        if cuda:
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        hidden_seq = [h_t]
        seq_size = inputs.size(1)

        # 0: normal 1: loss 2: repeat 3: out-of-order
        st = 0

        for t in range(seq_size):
            normal_max = self.state_tatble[st][0] + 0
            loss_max = self.state_tatble[st][1] + normal_max
            repeat_max = self.state_tatble[st][2] + loss_max
            out_of_order_max = self.state_tatble[st][3] + repeat_max
            rate = torch.rand(size=(1,)).cuda()

            if rate.le(normal_max)[0] or train is False:
                # print('normal ', train)
                x = inputs[:, t, :].t()

                i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)

                f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t + self.b_hf)

                c = torch.tanh(self.w_ic @ x + self.b_ic + self.w_hc @ h_t + self.b_hc)

                o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t + self.b_ho)

                c_next = f * c_t + i * c
                h_next = o * torch.tanh(c_next)
                c_next_t = c_next.t().unsqueeze(0)
                h_next_t = h_next.t().unsqueeze(0)
                hidden_seq.append(h_next_t)
                c_t = c_next
                h_t = h_next
                st = 0
            elif rate.le(loss_max)[0]:
                #print('loss ', train)
                st = 1
            elif rate.le(repeat_max)[0]:
                # print('repeat ', train)
                x = inputs[:, t, :].t()

                for _ in range(2):
                    i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)

                    f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t + self.b_hf)

                    c = torch.tanh(self.w_ic @ x + self.b_ic + self.w_hc @ h_t + self.b_hc)

                    o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t + self.b_ho)

                    c_next = f * c_t + i * c
                    h_next = o * torch.tanh(c_next)
                    c_next_t = c_next.t().unsqueeze(0)
                    h_next_t = h_next.t().unsqueeze(0)
                    hidden_seq.append(h_next_t)
                    c_t = c_next
                    h_t = h_next
                    st = 2
            else:
                # print('out-of-order ', train)
                x = inputs[:, t, :].t()

                x = x + 8 * random.randint(1, self.MTU // 8)

                x[x > self.MTU] = self.MTU

                i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t + self.b_hi)

                f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t + self.b_hf)

                c = torch.tanh(self.w_ic @ x + self.b_ic + self.w_hc @ h_t + self.b_hc)

                o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t + self.b_ho)

                c_next = f * c_t + i * c
                h_next = o * torch.tanh(c_next)
                c_next_t = c_next.t().unsqueeze(0)
                h_next_t = h_next.t().unsqueeze(0)
                hidden_seq.append(h_next_t)
                c_t = c_next
                h_t = h_next
                st = 3

        # hidden_seq = torch.cat(hidden_seq, dim=0)

        out = hidden_seq[-1].squeeze()
        out = self.fc(out)
        out = self.logsoftmax(out)

        # return hidden_seq, (h_next_t, c_next_t)
        return out
