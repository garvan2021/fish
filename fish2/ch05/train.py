# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# hyper-parameter set up
batch_size = 10
wordvec_size = 100
hidden_size = 100 # RNN hidden vector's element numbers
time_size = 5 # RNN expansion size
lr = 0.1
max_epoch = 100

# load train data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000 # reduce test dataset
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1] # input
ts = corpus[1:] # output(supervised label)

# generate models
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()
