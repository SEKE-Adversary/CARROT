#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class LSTMEncoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, brnn=True):
        
        super(LSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = brnn
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                            n_layers, dropout=drop_prob, bidirectional=brnn)
        
    def forward(self, input, hidden=None):
        return self.lstm(input, hidden)


class LSTMClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, encoder, hidden_dim, num_classes, max_len, dropout_p=0.3, attn=False):
        
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = encoder
        self.hidden_dim = hidden_dim * 2 if self.encoder.bidirectional else hidden_dim
        self.classify = nn.Linear(self.hidden_dim, num_classes)
        self.Dropout = nn.Dropout(dropout_p)
        self.max_len = max_len
        self.attn = attn
        if self.attn:
            self.W = nn.Parameter(torch.Tensor(np.zeros((self.hidden_dim, 1))))
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))
        
    def encode(self, inputs):
        
        emb = self.embedding(inputs)        
        outputs, hidden = self.encoder(emb)
        # outputs: (T, B, H * direc)
        # attention
        if self.attn:
            M = nn.Tanh()(outputs)
            M = M.permute([1, 0, 2])
            M = torch.reshape(M, [-1, self.hidden_dim])
            alpha = torch.mm(M, self.W)
            alpha = torch.reshape(alpha, [-1, self.max_len, 1])
            alpha = nn.Softmax(dim=1)(alpha)
            A = outputs.permute([1, 2, 0])
            r = torch.bmm(A, alpha)
            r = torch.squeeze(r)    # (B, H, 1)
            h_star = nn.Tanh()(r)
            drop = self.Dropout(h_star)
        else:
            drop = torch.mean(outputs, dim=0)
        return drop, emb
        
    def forward(self, x1, x2):
        
        vec1, emb1 = self.encode(x1)
        vec2, emb2 = self.encode(x2)
        abs_dist = torch.abs(torch.add(vec1, -vec2))

        logits = self.classify(abs_dist)
        return logits, (emb1, emb2)

    
    def prob(self, x1, x2):
        
        logits = self.forward(x1, x2)[0]
        prob = nn.Softmax(dim=1)(logits)
        return prob
        
    
    def grad(self, x1, x2, labels, loss_fn):
        
        # eg. loss_fn = nn.CrossEntropyLoss()
        # remove dropout
        savep1 = self.encoder.lstm.dropout
        savep2 = self.Dropout.p
        self.encoder.lstm.dropout = 0
        self.Dropout.p = 0
        self.zero_grad()
        logits, emb = self.forward(x1, x2)
        emb[0].retain_grad()
        emb[1].retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        # recover dropout
        self.encoder.lstm.dropout = savep1
        self.Dropout.p = savep2
        # emb.grad (T, B, Emb)
        return (emb[0].grad.permute([1, 0, 2]),
                emb[1].grad.permute([1, 0, 2]))

class GRUEncoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, brnn=True):
        
        super(GRUEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = brnn
        self.gru = nn.GRU(embedding_dim, hidden_dim, 
                          n_layers, dropout=drop_prob, bidirectional=brnn)
        
    def forward(self, input, hidden=None):
        return self.gru(input, hidden)


class GRUClassifier(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, encoder, hidden_dim, num_classes, max_len, dropout_p=0.3, attn=False):
        
        super(GRUClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = encoder
        self.hidden_dim = hidden_dim * 2 if self.encoder.bidirectional else hidden_dim
        self.classify = nn.Linear(self.hidden_dim, num_classes)
        self.Dropout = nn.Dropout(dropout_p)
        self.max_len = max_len
        self.attn = attn
        if self.attn:
            self.W = nn.Parameter(torch.Tensor(np.zeros((self.hidden_dim, 1))))
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))
        
    def encode(self, inputs):
        
        emb = self.embedding(inputs)        
        outputs, hidden = self.encoder(emb)
        # outputs: (T, B, H * direc)
        # attention
        if self.attn:
            M = nn.Tanh()(outputs)
            M = M.permute([1, 0, 2])
            M = torch.reshape(M, [-1, self.hidden_dim])
            alpha = torch.mm(M, self.W)
            alpha = torch.reshape(alpha, [-1, self.max_len, 1])
            alpha = nn.Softmax(dim=1)(alpha)
            A = outputs.permute([1, 2, 0])
            r = torch.bmm(A, alpha)
            r = torch.squeeze(r)    # (B, H, 1)
            h_star = nn.Tanh()(r)
            drop = self.Dropout(h_star)
        else:
            drop = torch.mean(outputs, dim=0)
        return drop, emb
        
    def forward(self, x1, x2):
        
        vec1, emb1 = self.encode(x1)
        vec2, emb2 = self.encode(x2)
        abs_dist = torch.abs(torch.add(vec1, -vec2))

        logits = self.classify(abs_dist)
        return logits, (emb1, emb2)

    
    def prob(self, x1, x2):
        
        logits = self.forward(x1, x2)[0]
        prob = nn.Softmax(dim=1)(logits)
        return prob
        
    
    def grad(self, x1, x2, labels, loss_fn):
        
        # eg. loss_fn = nn.CrossEntropyLoss()
        # remove dropout
        savep1 = self.encoder.gru.dropout
        savep2 = self.Dropout.p
        self.encoder.gru.dropout = 0
        self.Dropout.p = 0
        self.zero_grad()
        logits, emb = self.forward(x1, x2)
        emb[0].retain_grad()
        emb[1].retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        # recover dropout
        self.encoder.gru.dropout = savep1
        self.Dropout.p = savep2
        # emb.grad (T, B, Emb)
        return (emb[0].grad.permute([1, 0, 2]),
                emb[1].grad.permute([1, 0, 2]))