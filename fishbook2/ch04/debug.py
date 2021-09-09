import sys
sys.path.append('..')
import numpy as np
from common import config
config.GPU = True
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

# super-parameter setting
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# load data
corpus, word_to_id, id_to_word = ptb.load_data('train') # corpus.shape:929589, corpus:array([ 0,  1,  2, ..., 39, 26, 24])
vocab_size = len(word_to_id) # 10000

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# generate model.etc.
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

x, t, = contexts, target
data_size = len(x)
max_iters = data_size // batch_size
self.eval_interval = eval_interval
model, optimizer = self.model, self.optimizer
total_loss = 0 
loss_count = 0 
epoch = 0

idx = np.random.permutation(np.arange(data_size))
x = x[idx]
t = t[idx]

#~ for iters in range(max_iters):
iters = 0
batch_x = x[iters * batch_size:(iters + 1) * batch_size]
batch_t = t[iters * batch_size:(iters + 1) * batch_size]

# calculate gradient, update parameter
loss = model.forward(batch_x, batch_t)

# TypeError: forward() missing 1 required positional argument: 'target'
trainer.fit(contexts, target, max_epoch, batch_size)

for iters in range(max_iters):
	batch_x = x[iters * batch_size:(iters + 1) * batch_size]
	batch_t = t[iters * batch_size:(iters + 1) * batch_size]
	# calculate gradient, update parameter
	loss = model.forward(batch_x, batch_t)
	model.backward()
	params, grads = remove_duplicate(model.params, model.grads)
#                if max_grad is not None:
#                    clip_grads(grads, max_grad)
	optimizer.update(params, grads)
	total_loss += loss
	loss_count += 1
