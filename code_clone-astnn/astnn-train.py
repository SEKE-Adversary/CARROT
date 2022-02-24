# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:25:41 2020

@author: DrLC
"""

import os
import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings('ignore')
import gzip, pickle, random



def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.LongTensor(labels).squeeze(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", default="0")
    args = parser.parse_args()
    root = '../data_clone/'
    categories = 1
    train_data = pd.read_pickle(root+'/train/blocks.pkl').sample(frac=1)
    val_data = pd.read_pickle(root+'/dev/blocks.pkl').sample(frac=1)
    test_data = pd.read_pickle(root+'/test/blocks.pkl').sample(frac=1)

    word2vec = Word2Vec.load(root+"/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    #============================ ADV training ============================== if you don't need adv-training, comment this part
    #adv_train_size = 500
    #adv_train_path = '../model_clone/astnn/stmt_insert_atk_cand40_iter20_relax100_500.advsamples.pkl.gz'
    #adv_train_save_path = '../model_clone/astnn_stmt_insert_adv/model.pt'
    #adv_train_size = 1000
    #adv_train_path = '../model_clone/astnn/uid_rename_atk_cand40_iter50_relax100_1500.advsamples.pkl.gz'
    #adv_train_save_path = '../model_clone/astnn_uid_rename_adv/model.pt'
    adv_train_size = 1500
    adv_train_path = '../model_clone/astnn/cross.advsamples.pkl.gz'
    adv_train_save_path = '../model_clone/astnn_cross_adv/model.pt'

    with gzip.open(adv_train_path, 'rb') as f:
        d = pickle.load(f)
        adv_raw1 = d['adv_raw1s']
        adv_raw2 = d['adv_raw2s']
        adv_label = d['adv_labels']
    tmp_idxs = random.sample(range(adv_train_size), adv_train_size)
    adv_raw1 = [adv_raw1[i] for i in tmp_idxs]
    adv_raw2 = [adv_raw2[i] for i in tmp_idxs]
    adv_label = [adv_label[i] for i in tmp_idxs]
    print("[Adversarial Training] adversarial sample number: %d" % len(adv_raw1), flush=True)

    from pycparser import c_parser
    parser = c_parser.CParser()
    xx1 = [parser.parse(" ".join(raw)) for raw in adv_raw1]
    xx2 = [parser.parse(" ".join(raw)) for raw in adv_raw2]
    yy = adv_label
    from utils import get_data
    ins1, las = get_data(xx1, yy, word2vec)
    ins2, las = get_data(xx2, yy, word2vec)
    print("[Adversarial Training] adversarial sample number after parsing: %d" % len(las), flush=True)

    id1_base = max(train_data['id1']) + 1
    id1 = [i+id1_base for i in range(len(ins1))]
    id2_base = id1[-1] + 1
    id2 = [i+id2_base for i in range(len(ins2))]
    adv_train_data_dict = {'0': id1, '1': id2, '2': las, '3': ins1, '4': ins2}
    adv_train_data = pd.DataFrame(adv_train_data_dict)
    adv_train_data.columns = ['id1', 'id2', 'label', 'code_x', 'code_y']
    print("[Adversarial Training] training set size before: %d" % len(train_data), flush=True)
    train_data = pd.concat([train_data, adv_train_data])
    print("[Adversarial Training] training set size after: %d" % len(train_data), flush=True)
    #============================ ADV training ==============================

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    EPOCHS = 5
    BATCH_SIZE = 32
    USE_GPU = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    # print(train_data)
    precision, recall, f1 = 0, 0, 0
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    best_model = model
    print('Start training...')
    for t in range(1, categories+1):
        train_data_t, test_data_t = train_data, test_data
        # training procedure
        for epoch in range(EPOCHS):
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, train_labels.cuda()

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)
                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()
    
                # calc training acc
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == train_labels).sum()
                total += len(train_labels)
                total_loss += loss.item()*len(train1_inputs)
            train_loss_.append(total_loss / total)
            train_acc_.append(total_acc.item() / total)
            # validation epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(val_data):
                batch = get_batch(val_data, i, BATCH_SIZE)
                i += BATCH_SIZE
                val1_inputs, val2_inputs, val_labels = batch
                if USE_GPU:
                    val1_inputs, val2_inputs, val_labels = val1_inputs, val2_inputs, val_labels.cuda()
    
                model.batch_size = len(val_labels)
                model.hidden = model.init_hidden()
                output = model(val1_inputs, val2_inputs)
    
                loss = loss_function(output, Variable(val_labels))
    
                # calc valing acc
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == val_labels).sum()
                total += len(val_labels)
                total_loss += loss.item()*len(val1_inputs)
            val_loss_.append(total_loss / total)
            val_acc_.append(total_acc.item() / total)
            end_time = time.time()
            if val_acc_[epoch] > best_acc:
                best_model = model
                best_acc = val_acc_[epoch]
                #============================ ADV training ============================== if you don't need adv-training, use the fisrt statement
                #torch.save(model.state_dict(), '../model_clone/astnn/model.pt')
                torch.save(model.state_dict(), adv_train_save_path)
                #============================ ADV training ============================== 
                print ("\tCURRENT BEST!")
            print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
                  % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                     train_acc_[epoch], val_acc_[epoch], end_time - start_time),
                  flush=True)
        
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        model = best_model
        
        print("Testing-%d..."%t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test1_inputs, test2_inputs)
    
            loss = loss_function(output, Variable(test_labels))
    
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.item() * len(test1_inputs)

            # calc testing acc
            predicts.extend(predicted.cpu().numpy())
            trues.extend(test_labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
        print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))