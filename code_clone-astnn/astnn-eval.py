# -*- coding: utf-8 -*-

import torch
from model import BatchProgramCC
from pycparser import c_parser
import numpy as np
parser = c_parser.CParser()
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from utils import get_data
import os
from dataset import OJ104
import tqdm
import argparse
from sklearn.metrics import precision_recall_fscore_support

def get_batch(inputs1, inputs2, labels, i, bs):
    x1 = inputs1[i: i+bs]
    x2 = inputs2[i: i+bs]
    y = labels[i: i+bs]
    return x1, x2, torch.LongTensor(y).cuda()

if __name__ == "__main__":

    root = '../data_clone/'
    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    EPOCHS = 15
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-gpu', required=True)
    opt = arg_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                           USE_GPU, embeddings).cuda()
    model.load_state_dict(torch.load('../model_clone/astnn/model.pt'))
    
    oj = OJ104(path="../data_clone/oj.pkl.gz")
    xx1, xx2, yy = [], [], []
    for i in tqdm.tqdm(range(oj.test.get_size())):
        b = oj.test.next_batch(1)
        raw = ""
        for t in b['raw1'][0]:
            raw += t + " "
        xx1.append(parser.parse(raw))
        raw = ""
        for t in b['raw2'][0]:
            raw += t + " "
        xx2.append(parser.parse(raw))
        yy.append(b['y'][0])
    
    ins1, las = get_data(xx1, yy, word2vec)
    ins2, las = get_data(xx2, yy, word2vec)
    
    i = 0
    total_loss = 0.0
    total = 0
    loss_function = torch.nn.CrossEntropyLoss()
    
    n_sum, n_correct = 0, 0
    
    preds = []
    trues = []
    
    print ("Testing...")
    while i < len(yy):    
        inputs1, inputs2, labels = get_batch(ins1, ins2, las, i, BATCH_SIZE)
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        
        outputs = model(inputs1, inputs2)
        loss = loss_function(outputs, Variable(labels))
        total_loss += loss.item() * len(labels)
        total += len(labels)
        _, predicted = torch.max(outputs.data, 1)
        
        preds.extend(predicted.cpu().numpy())
        trues.extend(labels.cpu().numpy())
    
        for idx in range(len(labels)):
            n_sum += 1
            if predicted[idx] == labels[idx]:
                n_correct += 1
                
        i += BATCH_SIZE
    
    print('acc %.3f' % (n_correct/n_sum))
    print('loss %.3f' % (total_loss / total))    
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='binary')
    print("(P, R, F1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))