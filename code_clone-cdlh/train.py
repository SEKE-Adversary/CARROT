# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:34:12 2021

@author: DrLC
"""

import argparse
import sys
import os
from dataset import OJ_AST_GRAPH
from treelstm import TreeLSTM

import torch
import torch.nn as nn
from torch import optim
import numpy, random

from sklearn.metrics import precision_recall_fscore_support
    
    
def gettensor(batch, device):
    
    return batch['graph1'].to(device), batch['graph2'].to(device), \
        torch.tensor(batch['y'], dtype=torch.long).to(device)
    

    
def trainEpochs(epochs, training_set, valid_set, device,
                batch_size=1, print_each=100, saving_path='./'):
    
    classifier.train()
    
    epoch = 0
    i = 0
    print_loss_total = 0
    n_batch = int(training_set.get_size() / batch_size)
    
    print('start training epoch ' + str(epoch + 1) + '....')
    
    while True:
        
        batch = training_set.next_batch(batch_size)
        if batch['new_epoch']:
            epoch += 1
            evaluate(valid_set, batch_size)
            classifier.train()
            torch.save(classifier.state_dict(), os.path.join(saving_path, str(epoch) + '.pt'))
            
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            
            if epoch == epochs:
                break
            i = 0
            print_loss_total = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        inputs1, inputs2, labels = gettensor(batch, device)

        optimizer.zero_grad()
        
        
        outputs = classifier(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print_loss_total += loss.item()

        if (i + 1) % print_each == 0: 
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (epoch + 1, (i + 1) / n_batch * 100, print_loss_avg))
            
        i += 1
        

def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        
            
def evaluate(dataset, batch_size=1):
    
    classifier.eval()
    testnum = 0
    testcorrect = 0
    preds = []
    trues = []
    
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
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-l2p', type=float, default=0)
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('--data', type=str, default="../data_clone/oj_ast_graph.pkl.gz")
    parser.add_argument('--save_dir', type=str, default="../model_clone/cdlh")
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=20)
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    vocab_size = 2000
    embedding_size = 128
    hidden_size = 128
    num_classes = 2
    
    drop_prob = 0.1
    batch_size = opt.bs
    n_epoch = opt.nepoch
    rand_seed = 1726
    
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    

    oj = OJ_AST_GRAPH(path=opt.data, vocab_size=vocab_size)
    training_set = oj.train
    valid_set = oj.dev
    test_set = oj.test
    
    classifier = TreeLSTM(num_vocabs=oj.get_vocab_size(),
                          x_size=embedding_size,
                          h_size=hidden_size,
                          n_class=num_classes,
                          dropout=drop_prob,
                          device=device).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
    criterion = nn.CrossEntropyLoss()

    
    trainEpochs(n_epoch, training_set, valid_set, device,
                saving_path=opt.save_dir, batch_size=batch_size)