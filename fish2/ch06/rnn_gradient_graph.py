import numpy as np
import matplotlib.pyplot as plt

N = 2 # size of mini-batch
H = 3 # dimension of hidden vector 
T = 20 # length of time series data

dh = np.ones((N, H))
np.random.seed(3) # fix random seed for recurrent
Wh = np.random.randn(H, H)

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh ** 2)) / N
    norm_list.append(norm)
