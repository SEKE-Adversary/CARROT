import json
import pickle
import time
import os
import random
import torch
import numpy
import copy

import treemc as Tree
from model import BatchProgramClassifier
from pycparser import c_parser
import numpy as np
parser = c_parser.CParser()
from prepare_data import get_blocks as func
from gensim.models.word2vec import Word2Vec
from torch import Tensor, LongTensor
from utils import get_data, get_batch, convert, getInputs
from dataset import Dataset


model_path = './saved_models/adv-3000.pt'
data_path = './data/adv/poj104_testtrue1000.json'
vocab_path = './data/adv/poj104_vocab.json'
save_path = './data/adv/poj104_testtrue1000-adv-train-3000.pkl'
#     save_path = './data/adv/poj104_adv_train_'
n_required = 1000


root = './data/'
word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

vocab_size = 5000
max_len = 500
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 104
EPOCHS = 15
BATCH_SIZE = 128
USE_GPU = True
MAX_TOKENS = word2vec.syn0.shape[0]
EMBEDDING_DIM = word2vec.syn0.shape[1]

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

classifier = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                               USE_GPU, embeddings).cuda()
classifier.load_state_dict(torch.load(model_path))
classifier.batch_size = 1
classifier.hidden = classifier.init_hidden()
classifier.eval()
print ("MODEL LOADED!")


raw, rep, tree, label = [], [], [], []
with open(data_path, "r") as f:
    for _line in f.readlines():
        _d = json.loads(_line.strip())
        raw.append(_d["raw"])
        rep.append(_d["rep"])
        if _d['tree'] is not None:
            tree.append(Tree.dict2PTNode(_d["tree"]))
        else:
            tree.append(None)
        label.append(_d["label"])
with open(vocab_path, "r") as f:
    _d = json.loads(f.readlines()[0].strip())
    idx2token = _d["idx2token"][:vocab_size]
#     print(len(idx2token))
token2idx = {}
for i, t in zip(range(vocab_size), idx2token):
    token2idx[t] = i
dataset = Dataset(seq=rep, raw=raw, tree=tree, label=label,
                  idx2token=idx2token, token2idx=token2idx,
                  max_len=max_len, vocab_size=vocab_size,
                  dtype={'fp': numpy.float32, 'int': numpy.int32})
print ("DATA LOADED!")




print ("TEST MODEL...")
loss_function = torch.nn.CrossEntropyLoss()
_b = dataset.next_batch(1)
print("Original program:")
print(" ".join(_b['raw'][0]))
_inputs, _labels = getInputs(_b, False)
# print (classifier(_inputs))
# print (classifier.forward(_inputs))
# print (classifier.prob(_inputs))
# print (torch.argmax(classifier.prob(_inputs), dim=1))
grad = classifier.grad(_inputs, torch.tensor(_labels).cuda(), loss_function)
idx = word2vec.vocab['i'].index
print("Grad of \"i\":")
print(grad[idx])
idx = word2vec.vocab['j'].index
print("Grad of \"j\":")
print(grad[idx])
idx = word2vec.vocab['int'].index
print("Grad of \"int\":")
print(grad[idx])
print ("TEST MODEL DONE!")

