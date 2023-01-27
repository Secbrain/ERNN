
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
from sklearn.metrics import *

cuda = True if torch.cuda.is_available() else False
np.random.seed(42)
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed_all(42)


seq_size = 128 #time step
input_size = 1 #feature size
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


model_name='./model_para_eval/0.6+0.6.pth'
model.load_state_dict(torch.load(model_name))
model.eval()
y_test=np.load('../data_demo/test-y.npy')
lu='../data_demo/'
instances=['test-x.npy', 'loss-0.2.npy', 'retra-0.2.npy', 'order-0.2.npy', 'lr-0.2.npy', 'lo-0.2.npy', 'ro-0.2.npy', 'lro-0.2.npy']
x=np.array(range(128))
for j3 in instances:
    x_test=np.load(lu+j3)
    x_test=np.expand_dims(x_test, 2)
    fenlei=np.zeros((16000))
    quan = torch.Tensor(x_test[:, :seq_size, :])
    for i1 in range(160):
        q3=torch.Tensor(quan[i1*100:(i1+1)*100])
        images = Variable(q3.view(-1, seq_size, input_size).cuda())
        outputs = model.forward(images, state=None, train=False)
        _, predicted = torch.max(outputs.data, 1)
        fenlei[i1*100:(i1+1)*100]=np.array(predicted.cpu())
    np.save('./fenlei/att='+j3+'.npy',fenlei)
    acc=accuracy_score(y_test,fenlei)
    f1=f1_score(y_test,fenlei,average='macro')
    print('='*3,j3,'='*3)
    print('acc:  ',acc)
    print('f1:   ',f1)

