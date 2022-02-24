# -*- coding: utf-8 -*-

import pandas as pd
import random
import argparse
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys, pickle, gzip


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2])
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    root = '../data/'
    train_data = pd.read_pickle(root+'train/blocks.pkl')
    val_data = pd.read_pickle(root + 'dev/blocks.pkl')
    test_data = pd.read_pickle(root+'test/blocks.pkl')

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")    # the additional one is "<unk>"/"<pad>" 
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    #============================ ADV training ============================== if you don't need adv-training, comment this part
    adv_train_size = 4000
    #adv_train_path = '../model/astnn/stmt_insert_atk_candi40_iter20.advsamples.pkl.gz'
    #adv_train_save_path = '../model/astnn_stmt_insert_adv/model.pt'
    #adv_train_path = '../model/astnn/uid_rename_atk_candi40_iter50.advsamples.pkl.gz'
    #adv_train_save_path = '../model/astnn_uid_rename_adv/model.pt'
    adv_train_path = '../model/astnn/cross.advsamples.pkl.gz'
    adv_train_save_path = '../model/astnn_cross_adv/model.pt'

    with gzip.open(adv_train_path, 'rb') as f:
        d = pickle.load(f)
        adv_raw = d['adv_raw']
        adv_label = d['adv_label']
    tmp_idxs = random.sample(range(adv_train_size), adv_train_size)
    adv_raw = [adv_raw[i] for i in tmp_idxs]
    adv_label = [adv_label[i] for i in tmp_idxs]
    print("[Adversarial Training] adversarial sample number: %d" % len(adv_raw), flush=True)

    from pycparser import c_parser
    parser = c_parser.CParser()
    xx = [parser.parse(" ".join(raw)) for raw in adv_raw]
    yy = adv_label
    from utils import get_data
    ins, las = get_data(xx, yy, word2vec)
    print("[Adversarial Training] adversarial sample number after parsing: %d" % len(ins), flush=True)

    id_base = max(train_data['id']) + 1
    adv_train_data_dict = {'0': [i+id_base for i in range(len(ins))], '1': ins, '2': las}
    adv_train_data = pd.DataFrame(adv_train_data_dict)
    adv_train_data.columns = ['id', 'code', 'label']
    print("[Adversarial Training] training set size before: %d" % len(train_data), flush=True)
    train_data = pd.concat([train_data, adv_train_data])
    print("[Adversarial Training] training set size after: %d" % len(train_data), flush=True)
    #============================ ADV training ==============================

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 104
    EPOCHS = 15
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
#     print(MAX_TOKENS)
#     exit()

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()
    
#     model.load_state_dict(torch.load('./saved_models/1.pt'))
    
    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

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
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    
    #============================ ADV training ============================== if you don't need adv-training, use the fisrt statement
    #torch.save(model.state_dict(), '../model/astnn/model.pt')
    torch.save(model.state_dict(), adv_train_save_path)
    #============================ ADV training ============================== 
    
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    print("Testing results(Loss):", total_loss / total)
    
