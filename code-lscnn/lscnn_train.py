#-*- coding: utf-8 -*-
"""
Created on Thu Jun 17 20:49:40 2021

@author: DrLC
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

    
def trainEpochs(epochs, training_set, valid_set, device,
                batch_size=32, print_each=100, plot_each=100, saving_path='./'):
    
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
            evaluate(valid_set, device, batch_size)
            classifier.train()
            torch.save(classifier.state_dict(), os.path.join(saving_path, str(epoch) + '.pt'))
            
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            
            if epoch == epochs:
                break
            i = 0
            print_loss_total = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        inputs, lens, labels = gettensor(batch, device)

        optimizer.zero_grad()
        
        
        outputs = classifier(inputs, lens)
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
            outputs = classifier(inputs, lens)
        
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
    parser.add_argument('--data', type=str, default="../data/oj.pkl.gz")
    parser.add_argument('--save_dir', type=str, default="../model/lscnn")
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--nepoch', type=int, default=30)
    
    
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
    n_epoch = opt.nepoch
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
    
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
    criterion = nn.CrossEntropyLoss()
    
    trainEpochs(n_epoch, training_set, valid_set, device,
                saving_path=opt.save_dir, batch_size=batch_size)
