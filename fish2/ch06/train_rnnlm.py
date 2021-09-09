import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity

from dataset import ptb
from rnnlm import Rnnlm

# setting hyper-parameter
batch_size = 20
wordvec_size = 100
hidden_size = 100 # RNN hidden vector's element number
time_size = 35 # RNN's expansion size 
lr = 20.0
max_epoch = 4
max_grad = 0.25

# load train data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# generate model
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 1.apply gradient crop to learn
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval = 20)
trainer.plot(ylim = (0, 500))

# 2.evaluation based on test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# 3.save parameter
model.save_params()
