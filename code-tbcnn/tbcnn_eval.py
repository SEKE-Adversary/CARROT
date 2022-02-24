# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:57:05 2021

@author: DrLC
"""

import argparse
import sys
import os
from dataset import OJ104_AST_GRAPH
from tbcnn import TBCNNClassifier

import torch
import torch.nn as nn
from torch import optim
import numpy, random
    
    
def gettensor(batch, device):
    
    return batch['graph'].to(device), torch.tensor(batch['y'], dtype=torch.long).to(device)
            
def evaluate(dataset, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        #print(batch)
        #break
        if batch['new_epoch']:
            break
        inputs, labels = gettensor(batch, device)
        
        with torch.no_grad():
            outputs = classifier(inputs)
        
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)

    print('eval_acc:  %.5f' % (float(testcorrect) * 100.0 / testnum))
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-l2p', type=float, default=0)
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('--data', type=str, default="../data/oj_ast_graph.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model/tbcnn/model.pt")
    parser.add_argument('--bs', type=int, default=32)
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    vocab_size = 5000
    embedding_size = 256
    hidden_size = 256
    n_layers = 1
    num_classes = 104
    
    drop_prob = 0.
    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    

    poj = OJ104_AST_GRAPH(path=opt.data,
                          vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    
    classifier = TBCNNClassifier(x_size=embedding_size,
                                 h_size=hidden_size,
                                 dropout=drop_prob,
                                 num_layers=n_layers,
                                 n_classes=num_classes,
                                 vocab_size=poj.get_vocab_size()).to(device)
    
    classifier.load_state_dict(torch.load(opt.model_path))

    
    evaluate(test_set, batch_size)
