# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:36:09 2021

@author: LENOVO
"""

import argparse
import os
from dataset import CODECHEF
from lscnn import LSCNNClassifier

import torch
import random
import numpy
from sklearn import metrics

    
    
    
def gettensor(batch, device):
    
    return (torch.tensor(batch['x'], dtype=torch.long).to(device),
            torch.tensor(batch['l'], dtype=torch.long).to(device),
            torch.tensor(batch['y'], dtype=torch.long).to(device))
        
            
def evaluate(dataset, device, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    trues = []
    preds = []
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, lens, labels = gettensor(batch, device)
        
        with torch.no_grad():
            outputs = classifier(inputs, lens)
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
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('--data', type=str, default="../data_defect/codechef.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model_defect/lscnn/model.pt")
    parser.add_argument('--bs', type=int, default=128)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 4
    vocab_size = 3000
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

    cc = CODECHEF(path=opt.data,
                  max_stmt_len=max_stmt_len,
                  max_stmt_cnt=max_stmt_cnt,
                  vocab_size=vocab_size)
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test
    
    classifier = LSCNNClassifier(n_class, cc.get_vocab_size(), embed_width,
                                 n_conv, conv_size, lstm_size, n_lstm,
                                 brnn, device).to(device)
    classifier.load_state_dict(torch.load(opt.model_path))
    
    evaluate(test_set, device, batch_size)