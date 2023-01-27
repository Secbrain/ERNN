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
from ernn import *

cuda = True if torch.cuda.is_available() else False


torch.manual_seed(2022)
if cuda:
    torch.cuda.manual_seed_all(2022)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#=== x: [sample, seq_size, input_size(feature_size)] ===#
lu='./dataset_dir/'
x=np.load('train_x.npy')
x_test=np.load('test_x.npy')
y=np.load('train_y.npy')
y_test=np.load('test_y.npy')


x=np.expand_dims(x, 2)
x_test=np.expand_dims(x_test, 2)
batch_size = 160
num_batch = int(y.shape[0] / batch_size)
num_epochs = 205

seq_size = 128
input_size = 1
hidden_size = 400
layer_size = 2
output_size = 16
Fm = [[1, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 0, 0],
      [1, 0, 0, 0]]

model = RLSTM(input_size, hidden_size, output_size, Fm)
if cuda:
    model.cuda()


criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
all_parameters = model.parameters()
weight_parameters = []
for pname, p in model.named_parameters():
    if p.size(0) != 1:
        weight_parameters.append(p)


optimizer = torch.optim.Adam([{'params': weight_parameters}], lr=learning_rate)
loss_list = []
it = 0
best_accuracy=0
best_model_weight = None
for epoch in range(num_epochs):
    np.random.seed(epoch)
    rand = np.arange(0, 64000, 1)
    np.random.shuffle(rand)
    for i in range(num_batch):
        images = torch.Tensor(x[rand[i * batch_size:(i + 1) * batch_size], :seq_size, :])
        labels = torch.Tensor(y[rand[i * batch_size:(i + 1) * batch_size]]).long()
        if cuda:
            images = Variable(images.view(-1, seq_size, input_size).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_size, input_size))
            labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if cuda:
            loss.cuda()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        it += 1
        if it % 200 == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                while True:
                    images = torch.Tensor(x_test[:, :seq_size, :])
                    labels = torch.Tensor(y_test[:]).long()
                    if cuda:
                        images = Variable(images.view(-1, seq_size, input_size).cuda())
                    else:
                        images = Variable(images.view(-1, seq_size, input_size))
                    outputs = model.forward(images, state=None, train=False)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    if cuda:
                        correct += (predicted.cpu() == labels.cpu()).sum()
                    else:
                        correct += (predicted == labels).sum()
                    break
                accuracy = 100 * torch.true_divide(correct, total)
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(it, loss.item(), accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_weight = copy.deepcopy(model.state_dict())
                    torch.save(best_model_weight, './model/best_model_weight-'+str(it)+'.pth')
                    print('save-------   '+str(it))


