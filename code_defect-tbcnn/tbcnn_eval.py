# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:57:05 2021

@author: DrLC
"""

import argparse
import os
from dataset import CODECHEF_AST_GRAPH
from tbcnn import TBCNNClassifier

import torch
import numpy, random
from sklearn import metrics
    
    
def gettensor(batch, device):
    
    return batch['graph'].to(device), torch.tensor(batch['y'], dtype=torch.long).to(device)
            
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
        inputs, labels = gettensor(batch, device)
        
        with torch.no_grad():
            outputs = classifier(inputs)
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
    parser.add_argument('--data', type=str, default="../data_defect/codechef_ast_graph.pkl.gz")
    parser.add_argument('--model_path', type=str, default="../model_defect/tbcnn/model.pt")
    parser.add_argument('--bs', type=int, default=32)
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    vocab_size = 3000
    embedding_size = 256
    hidden_size = 256
    n_layers = 1
    num_classes = 4
    
    drop_prob = 0.
    batch_size = opt.bs
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    

    cc = CODECHEF_AST_GRAPH(path=opt.data,
                            vocab_size=vocab_size)
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test
    
    classifier = TBCNNClassifier(x_size=embedding_size,
                                 h_size=hidden_size,
                                 dropout=drop_prob,
                                 num_layers=n_layers,
                                 n_classes=num_classes,
                                 vocab_size=cc.get_vocab_size()).to(device)
    
    classifier.load_state_dict(torch.load(opt.model_path))

    
    evaluate(test_set, device, batch_size)