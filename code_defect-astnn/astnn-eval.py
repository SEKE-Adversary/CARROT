# -*- coding: utf-8 -*-

import torch
from model import BatchProgramClassifier
from pycparser import c_parser
import numpy as np
parser = c_parser.CParser()
from gensim.models.word2vec import Word2Vec
from torch import LongTensor
from utils import get_data, get_batch
import os
from dataset import CodeChef
import tqdm
import argparse
from sklearn import metrics

root = '../data_defect/'
word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 4
EPOCHS = 15
BATCH_SIZE = 1
USE_GPU = True
MAX_TOKENS = word2vec.syn0.shape[0]
EMBEDDING_DIM = word2vec.syn0.shape[1]

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-gpu', default="0")
opt = arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                               USE_GPU, embeddings).cuda()
model.load_state_dict(torch.load('../model_defect/astnn/model.pt'))

oj = CodeChef()
xx, yy, idid = [], [], []
for i in tqdm.tqdm(range(oj.test.get_size())):
    b = oj.test.next_batch(1)
    raw = ""
    for t in b['raw'][0]:
        raw += t + " "
    xx.append(parser.parse(raw))
    yy.append(b['y'][0])
    idid.append(b['id'][0])

ins, las = get_data(xx, yy, word2vec)

i = 0
total_loss = 0.0
total = 0
loss_function = torch.nn.CrossEntropyLoss()

n_sum, n_correct = 0, 0
trues = []
preds = []

while i < len(yy):    
    inputs, labels = get_batch(ins, las, i, BATCH_SIZE)
    model.batch_size = len(labels)
    model.hidden = model.init_hidden()
    
    outputs = model(inputs)
    loss = loss_function(outputs, LongTensor(labels).cuda())
    total_loss += loss.item() * len(labels)
    total += len(labels)
    outputs = torch.max(outputs, 1)[1]

    for idx in range(len(labels)):
        n_sum += 1
        if outputs[idx] == labels[idx]:
            n_correct += 1
    trues.extend(labels)
    preds.extend(outputs.cpu().numpy())
    for y, p, _id in zip (labels, outputs.cpu().numpy(), idid[i*BATCH_SIZE: (i+1)*BATCH_SIZE]):
            print (_id, p)
    i += BATCH_SIZE

print('acc %.5f' % (n_correct/n_sum))
print('loss %.5f' % (total_loss / total))
print('eval precision:  %.5f' % metrics.precision_score(trues, preds, average='macro'))
print('eval recall:  %.5f' % metrics.recall_score(trues, preds, average='macro'))
print('eval f1:  %.5f' % metrics.f1_score(trues, preds, average='macro'))
    