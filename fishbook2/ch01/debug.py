import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

# set hyperparameter
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size = 2, hidden_size = hidden_size, output_size = 3)
optimizer = SGD(lr = learning_rate)

# learning variable
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

epoch = 0
# scramble data
idx = np.random.permutation(data_size)
x = x[idx]
t = t[idx]

iters = 0
batch_x = x[iters * batch_size:(iters + 1) * batch_size]
batch_t = t[iters * batch_size:(iters + 1) * batch_size]

loss = model.forward(batch_x, batch_t)

dout = 1
dout = self.loss_layer.backward(dout)


#~ model.backward()
#~ optimizer.update(model.params, model.grads)

#~ total_loss += loss
#~ loss_count += 1
