# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 11:56:46 2021

@author: DrLC
"""

import argparse
import sys
import os
from dataset import OJ_AST_GRAPH
from tbcnn import TBCNNClassifier
from sklearn.metrics import precision_recall_fscore_support

import torch
import numpy, random
    
    
def gettensor(batch, device):
    
    return batch['graph1'].to(device), batch['graph2'].to(device), \
        torch.tensor(batch['y'], dtype=torch.long).to(device)
        
            
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
        inputs1, inputs2, labels = gettensor(batch, device)
        
        with torch.no_grad():
            outputs = classifier(inputs1, inputs2)
        
            predicted = torch.argmax(outputs, dim=1)
            res = predicted == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    print('eval_acc:  %.2f' % (float(testcorrect) * 100.0 / testnum))
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, average='binary')
    print("(P, R, F1) = (%.4f, %.4f, %.4f)" % (precision, recall, f1))
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('--data', type=str, default="../data_clone/oj_ast_graph.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model_clone/tbcnn/model.pt")
    parser.add_argument('--bs', type=int, default=16)
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    vocab_size = 2000
    embedding_size = 256
    hidden_size = 256
    num_classes = 2
    
    drop_prob = 0.
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    

    oj = OJ_AST_GRAPH(path=opt.data, vocab_size=vocab_size)
    training_set = oj.train
    valid_set = oj.dev
    test_set = oj.test
    
    classifier = TBCNNClassifier(x_size=embedding_size,
                                 h_size=hidden_size,
                                 dropout=drop_prob,
                                 n_classes=num_classes,
                                 vocab_size=oj.get_vocab_size()).to(device)
    classifier.load_state_dict(torch.load(opt.model_path))

    
    evaluate(test_set, device, opt.bs)