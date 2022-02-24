# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:29:07 2021

@author: LENOVO
"""

import argparse
import os
from dataset import OJ104

import torch
import torch.nn as nn
from torch import optim
import random
import numpy
        

def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
            
def evaluate(dataset, device, batch_size=128):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    
    while True:
        
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        
        with torch.no_grad():
            outputs = classifier(batch['x'])
            labels = torch.tensor(batch['y'], dtype=torch.long).to(device)
            res = torch.argmax(outputs, dim=1) == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)

    print('eval_acc:  %.5f' % (float(testcorrect) * 100.0 / testnum))
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('--data', type=str, default="../data/oj.pkl.gz")
    parser.add_argument('--model_dir', type=str, default="../model/codebert/model")
    parser.add_argument('--bs', type=int, default=16)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 104

    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    poj = OJ104(path=opt.data)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    
    # import transformers after gpu selection
    from codebert import CodeBERTClassifier
    
    classifier = CodeBERTClassifier(model_path=opt.model_dir,
                                    num_labels=n_class,
                                    device=device).to(device)
    #print(classifier.model)    
    print(classifier.vocab)
    evaluate(test_set, device, batch_size)
