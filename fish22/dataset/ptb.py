# coding: utf-8
import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportErros:
    raise ImportError('Use Python3!')
import numpy as np

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))

def _download(file_name):
    file_path = dataset_dir + '/' + file_name
    if os.path.exists(file_path):
        return

    print('Downloading ' + file_name + ' ... ')

    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
    except urllib.error.URLError:
        import ssl
        ssl.create_default_https_context = ssl.create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)

    print('Done')

def load_data(data_type='train'):
    if data_type == 'val': data_type = 'valid'
    # word_to_id, id_to_word = load_vocab()
    word_to_id = {}
    id_to_word = {}

    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

def load_vocab():
    word_to_id = {}
    id_to_word = {}
    file_name = key_file['train']
    file_path = dataset_dir + '/' + file_name

    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            idx = len(word_to_id)
            word_to_id[word] = idx
            id_to_word[idx] = word

    #with open (vocab_path, 'wb') as f:
    #   pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word
