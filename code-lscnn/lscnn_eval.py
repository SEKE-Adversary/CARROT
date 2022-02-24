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
    
    
    
    
def gettensor(batch, device):
    
    return (torch.tensor(batch['x'], dtype=torch.long).to(device),
            torch.tensor(batch['l'], dtype=torch.long).to(device),
            torch.tensor(batch['y'], dtype=torch.long).to(device))
        
            
def evaluate(dataset, device, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, lens, labels = gettensor(batch, device)
        
        with torch.no_grad():
            #print("in:", inputs.size())
            outputs = classifier(inputs, lens)
            #print("out:", outputs.size())

            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)

    print('eval_acc:  %.5f' % (float(testcorrect) * 100.0 / testnum))
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('--data', type=str, default="../data/oj.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model/lscnn/model.pt")
    parser.add_argument('--bs', type=int, default=128)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 104
    vocab_size = 5000
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
