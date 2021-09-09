# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # ????????????????
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# ????
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# ???????????????? 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet()
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
batch_mask = np.random.choice(trainer.train_size, trainer.batch_size)
x_batch = trainer.x_train[batch_mask]
t_batch = trainer.t_train[batch_mask]
grads = trainer.network.gradient(x_batch, t_batch)
trainer.optimizer.update(self.network.params, grads)
#up to here







# done
x1 = trainer.network.layers['Conv1'].forward(x_batch)
x2 = trainer.network.layers['Relu1'].forward(x1)
x3 = trainer.network.layers['Pool1'].forward(x2)
x4 = trainer.network.layers['Affine1'].forward(x3)
x5 = trainer.network.layers['Relu2'].forward(x4)
x6 = trainer.network.layers['Affine2'].forward(x5)
y = trainer.network.lastLayer = SoftmaxWithLoss()