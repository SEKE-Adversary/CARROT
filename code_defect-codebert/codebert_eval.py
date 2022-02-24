# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:29:07 2021

@author: LENOVO
"""

import argparse
import os
from dataset import CODECHEF

import torch
import random
import numpy
from sklearn import metrics
        
            
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
        
        with torch.no_grad():
            labels = torch.tensor(batch['y'], dtype=torch.long).to(device)
            outputs = classifier(batch['x'])
            pred = torch.argmax(outputs, dim=1)
            res = pred == labels
            testcorrect += torch.sum(res)
            testnum += len(labels)
            trues.extend(labels.cpu().numpy())
            preds.extend(pred.cpu().numpy())
                

    print('eval_acc:  %.5f' % (float(testcorrect) * 100.0 / testnum))
    print('eval precision:  %.5f' % metrics.precision_score(trues, preds, average='macro'))
    print('eval recall:  %.5f' % metrics.recall_score(trues, preds, average='macro'))
    print('eval f1:  %.5f' % metrics.f1_score(trues, preds, average='macro'))
    
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="-1")
    parser.add_argument('--data', type=str, default="../data_defect/codechef.pkl.gz")
    parser.add_argument('--model_dir', type=str, default="../model_defect/codebert/model")
    parser.add_argument('--bs', type=int, default=16)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 4

    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    cc = CODECHEF(path=opt.data)
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test
    
    # import transformers after gpu selection
    from codebert import CodeBERTClassifier
    
    classifier = CodeBERTClassifier(model_path=opt.model_dir,
                                    num_labels=n_class,
                                    device=device).to(device)
    
    evaluate(test_set, device, batch_size)