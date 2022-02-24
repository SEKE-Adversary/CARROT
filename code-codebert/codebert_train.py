# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:42:42 2021

@author: DrLC
"""

import argparse
import os
from dataset import OJ104

import torch
import torch.nn as nn
from torch import optim
import random
import numpy

    
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
            
            classifier.model.save_pretrained(os.path.join(saving_path, str(epoch) + ".pt"))
            classifier.tokenizer.save_pretrained(os.path.join(saving_path, str(epoch) + ".pt"))
            
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            
            if epoch == epochs:
                break
            i = 0
            print_loss_total = 0
            print('start training epoch ' + str(epoch + 1) + '....')

        optimizer.zero_grad()
        
        ouputs, loss = classifier(batch['x'], batch['y'])
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
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-l2p', type=float, default=0)
    parser.add_argument('-lrdecay', action='store_true')
    parser.add_argument('--data', type=str, default="../data/oj.pkl.gz")
    parser.add_argument('--pretrain', type=str, default="../model_codebert_base_mlm")
    parser.add_argument('--save_dir', type=str, default="../model/codebert")
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--nepoch', type=int, default=10)
    
    
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    n_class = 104

    batch_size = opt.bs
    n_epoch = opt.nepoch
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
    
    classifier = CodeBERTClassifier(model_path=opt.pretrain,
                                    num_labels=n_class,
                                    device=device).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
    
    trainEpochs(n_epoch, training_set, valid_set, device,
                saving_path=opt.save_dir, batch_size=batch_size)