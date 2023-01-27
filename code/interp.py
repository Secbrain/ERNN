
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
from sklearn.metrics import *
from ernn import *
import time
import matplotlib.pyplot as plt
from torch.optim import Adam
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

cuda = True if torch.cuda.is_available() else False
np.random.seed(42)
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed_all(42)


seq_size = 128
input_size = 1
hidden_size = 400
layer_size = 2
output_size = 16

Fm = [[1, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 0, 0]]

lujing='./'

model = RLSTM(input_size, hidden_size, output_size, Fm)
if cuda:
    model.cuda()
    
model.load_state_dict(torch.load('./XXX.pth'))
model.eval()
y_test=np.load('./dataset/test-y.npy')
lu='./shaper/'
instances=['test-128.npy', 'loss-0.2.npy', 'retra-0.2.npy', 'order-0.2.npy', 'lr-0.2.npy', 'lo-0.2.npy', 'ro-0.2.npy', 'lro-0.2.npy']
x=np.array(range(128))
baseline = torch.zeros(10,128,1).cuda()
ig = IntegratedGradients(model)
for j3 in instances:
    x_test=np.load(lu+j3)
    x_test=np.expand_dims(x_test, 2)
    jieguo=np.zeros((16000,128))
    quan = torch.Tensor(x_test[:, :seq_size, :])
    for i1 in range(1600):
        q3=torch.Tensor(quan[i1*10:(i1+1)*10])
        images = Variable(q3.view(-1, seq_size, input_size).cuda())
        outputs = model.forward(images, state=None, train=False)
        _, predicted = torch.max(outputs.data, 1)
        attributions, delta = ig.attribute(images[:,:,:], baseline, target=0, return_convergence_delta=True)
        nn=attributions.cpu().detach().numpy().copy()
        jieguo[i1*10:(i1+1)*10]=nn[:,:,0]
    np.save('./attrib/att='+j3,jieguo)

