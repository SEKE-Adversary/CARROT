# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:36:09 2021

@author: LENOVO
"""

import argparse
import sys
import os
from dataset import OJ104
from lscnn import LSCNNClassifier

import torch
import torch.nn as nn
from torch import optim
import random
import numpy

from sklearn import metrics
    
    
    
    
def gettensor(batch, device):
    
    return (torch.tensor(batch['x1'], dtype=torch.long).to(device),
            torch.tensor(batch['x2'], dtype=torch.long).to(device),
            torch.tensor(batch['l1'], dtype=torch.long).to(device),
            torch.tensor(batch['l2'], dtype=torch.long).to(device),
            torch.tensor(batch['y'], dtype=torch.long).to(device))
        
            
def evaluate(dataset, device, batch_size=128):
    
    preds = []
    trues = []
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs1, inputs2, lens1, lens2, labels = gettensor(batch, device)
        
        with torch.no_grad():
            outputs = classifier(inputs1, inputs2, lens1, lens2)
        
            predicted = torch.argmax(outputs, dim=1)
            res = predicted == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    print('eval_acc:  %.5f' % (float(testcorrect) * 100.0 / testnum))
    print('eval precision:  %.5f' % metrics.precision_score(trues, preds, average='binary'))
    print('eval recall:  %.5f' % metrics.recall_score(trues, preds, average='binary'))
    print('eval f1:  %.5f' % metrics.f1_score(trues, preds, average='binary'))
    p, r, f, _ = metrics.precision_recall_fscore_support(trues, preds, average='binary')
    print(p, r, f)
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('--data', type=str, default="../data_clone/oj.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model_clone/lscnn/model.pt")
    parser.add_argument('--bs', type=int, default=128)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 2
    vocab_size = 2000
    embed_width = 512
    n_conv = 300
    conv_size = 5
    lstm_size = 400
    n_lstm = 1
    max_stmt_cnt = 40
    max_stmt_len = 20
    brnn = True

    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = OJ104(path=opt.data,
                max_stmt_len=max_stmt_len,
                max_stmt_cnt=max_stmt_cnt,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    
    classifier = LSCNNClassifier(n_class, vocab_size, embed_width,
                                 n_conv, conv_size, lstm_size, n_lstm,
                                 brnn, device).to(device)
    classifier.load_state_dict(torch.load(opt.model_path))
    
    evaluate(test_set, device, batch_size)