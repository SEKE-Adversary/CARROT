# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:10:27 2020

@author: DrLC
"""

import argparse
import os
from dataset import CodeChef
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUClassifier, GRUEncoder

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

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

def evaluate(dataset, batch_size=1):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    trues = []
    preds = []
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels = gettensor(batch, batchfirst=False)
        
        with torch.no_grad():
            outputs = classifier(inputs)[0]
            pred = torch.argmax(outputs, dim=1)
            res = pred == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
            trues.extend(labels.cpu().numpy())
            preds.extend(pred.cpu().numpy())
        for y, p, _id in zip (labels.cpu().numpy(), pred.cpu().numpy(), batch['id']):
            print (_id, p)
                

    print('eval_acc:  %.5f' % (float(testcorrect) * 100.0 / testnum))
    print('eval precision:  %.5f' % metrics.precision_score(trues, preds, average='macro'))
    print('eval recall:  %.5f' % metrics.recall_score(trues, preds, average='macro'))
    print('eval f1:  %.5f' % metrics.f1_score(trues, preds, average='macro'))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    parser.add_argument('-attn', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")

    vocab_size = 3000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 4
    max_len = 300
    dropprob = 0.

    cc = CodeChef(max_len=max_len,
                  vocab_size=vocab_size)
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test
    
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers, drop_prob=dropprob)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len, attn=opt.attn, dropout_p=dropprob).cuda()
    classifier.load_state_dict(torch.load('../model_defect/lstm/model-13.pt'))
    
    #enc = GRUEncoder(embedding_size, hidden_size, n_layers, drop_prob=dropprob)
    #classifier = GRUClassifier(vocab_size, embedding_size, enc,
    #                            hidden_size, num_classes, max_len, attn=opt.attn, dropout_p=dropprob).cuda()
    #classifier.load_state_dict(torch.load('../model_defect/gru/model-15.pt'))

    print()
    print()
    print('eval on test set...')
    evaluate(test_set)