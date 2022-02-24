# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:10:27 2020

@author: DrLC
"""

import argparse
import os
from dataset import OJ104
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUClassifier, GRUEncoder

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

def gettensor(batch, batchfirst=False):
    
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).cuda(), \
                                    torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
#         inputs_pos = [[pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)] for inst in inputs]
#         inputs_pos = torch.tensor(inputs_pos, dtype=torch.long).cuda()
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels

def evaluate(dataset, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels = gettensor(batch, batchfirst=(_model == 'Transformer'))
        
        with torch.no_grad():
            outputs = classifier(inputs)[0]
        
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)

    print('eval_acc:  %.2f' % (float(testcorrect) * 100.0 / testnum))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-attn', action='store_true')
    
    
    opt = parser.parse_args()
    _model = opt.model
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")

    vocab_size = 5000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 104
    max_len = 300

    poj = OJ104(path='../data/oj.pkl.gz',
                max_len=max_len,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    
    if _model == 'LSTM':
        enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
        classifier = LSTMClassifier(vocab_size, embedding_size, enc, hidden_size, num_classes, max_len, attn=opt.attn).cuda()
        #classifier.load_state_dict(torch.load('../model/lstm/model-14.pt'))
        #classifier.load_state_dict(torch.load('../model/mhm_lstm_model_15.pt'))
        #classifier.load_state_dict(torch.load('../model/lstm_uid_rename_adv/15.pt'))
        #classifier.load_state_dict(torch.load('../model/lstm_stmt_insert_adv/15.pt'))
        classifier.load_state_dict(torch.load('../model/lstm_cross_adv/15.pt'))
    elif _model == 'GRU':
        enc = GRUEncoder(embedding_size, hidden_size, n_layers)
        classifier = GRUClassifier(vocab_size, embedding_size, enc, hidden_size, num_classes, max_len, attn=opt.attn).cuda()
        classifier.load_state_dict(torch.load('../model/gru/model-15.pt'))
    elif _model == 'Transformer':
        exit()

    print()
    print()
    print('eval on test set...')
    evaluate(test_set)