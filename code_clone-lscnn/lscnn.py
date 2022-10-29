# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:25:29 2021

@author: DrLC
"""

import torch
import torch.nn as nn

class LSCNNClassifier(torch.nn.Module):
    
    def __init__(self,  n_class, vocab_size, embed_width, n_conv, conv_size,
                 lstm_size, n_lstm, bilstm=True, device=None):
        
        super(LSCNNClassifier, self).__init__()
        
        self.embed_width = embed_width
        self.device = device
        self.n_conv = n_conv
        self.lstm_size = lstm_size * 2 if bilstm else lstm_size
        
        self.vocab_size = vocab_size
        self.embed_width = self.x_size = embed_width
        self.embed = nn.Embedding(vocab_size, embed_width)
        self.conv = nn.Conv1d(embed_width, n_conv, conv_size, stride=1, padding=(conv_size-1)//2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.lstm = nn.LSTM(n_conv, lstm_size, n_lstm, bidirectional=bilstm)
        self.classify = nn.Linear(self.lstm_size, n_class)
        
    def _forward(self, x, l):
        
        bs, stmt_cnt, stmt_len = x.shape
        assert bs == l.shape[0] and stmt_cnt == l.shape[1]
        
        # Embedding
        embedding = self.embed(x)
        embedding = embedding.reshape([bs * stmt_cnt, stmt_len, self.embed_width])
        embedding = embedding.transpose(1, 2)
        # [ bs * stmt_cnt x embed_width x stmt_len ]
        
        # Convolution
        convolution = self.conv(embedding)
        _l = l.reshape([bs * stmt_cnt])
        conv_mask = (torch.arange(stmt_len).to(self.device)[None, :] < _l[:, None])
        conv_mask = torch.stack([conv_mask] * self.n_conv, 1)
        convolution = convolution.masked_fill(~conv_mask, float("-inf"))
        # [ bs * stmt_cnt x n_conv x stmt_len ]
        
        # Pooling
        pooling = self.pool(convolution)
        pooling = pooling.squeeze()
        pooling = pooling.reshape([bs, stmt_cnt, self.n_conv])
        pool_mask = torch.isinf(pooling)
        pooling = pooling.masked_fill(pool_mask, 0)
        pooling = pooling.transpose(0, 1)
        # [ stmt_cnt x bs x n_conv ]
        
        # LSTM
        h, _ = self.lstm(pooling)
        h = h.transpose(0, 1)
        h_mask = l == -1
        h_mask = torch.stack([h_mask] * self.lstm_size, 2)
        h = h.masked_fill(h_mask, float("-inf"))
        h = h.transpose(1, 2)
        # [ bs x stmt_cnt x lstm_size ]
        
        # Pooling
        pooling2 = self.pool(h)
        pooling2 = pooling2.squeeze()
        # [ bs x lstm_size ]
        
        pooling2 = pooling2.reshape([bs, -1])
        return pooling2
    
    def forward(self, x1, x2, l1, l2):
        
        vec1 = self._forward(x1, l1)
        vec2 = self._forward(x2, l2)
        # size: [bs, lstm_size]
        abs_dist = torch.abs(torch.add(vec1, -vec2))
        logits = self.classify(abs_dist)
        return logits

    def prob(self, x1, x2, l1, l2):

        logits = self.forward(x1, x2, l1, l2)
        prob = nn.Softmax(dim=-1)(logits)
        return prob

    def grad(self, x1, x2, l1, l2, labels, loss_fn):
        
        vec2 = self._forward(x2, l2)
        vec2_const = vec2.clone().detach()

        self.zero_grad()
        vec1 = self._forward(x1, l1)
        abs_dist = torch.abs(torch.add(vec1, -vec2_const))
        logits = self.classify(abs_dist)

        self.embed.weight.retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()

        return self.embed.weight.grad
    
if __name__ == "__main__":
    
    from dataset import OJ104
    
    n_class = 2
    vocab_size = 2000
    embed_width = 512
    n_conv = 300
    conv_size = 5
    lstm_size = 400
    n_lstm = 1
    
    bs = 3
    stmt_len = 20
    stmt_cnt = 40
    
    oj = OJ104(path='../data_clone/oj.pkl.gz', max_stmt_len=stmt_len,
               max_stmt_cnt=stmt_cnt, vocab_size=vocab_size)
    model = LSCNNClassifier(n_class, vocab_size, embed_width, n_conv, conv_size, lstm_size, n_lstm)
    
    b = oj.train.next_batch(bs)
    
    _x1 = torch.tensor(b['x1'], dtype=torch.long)
    _l1 = torch.tensor(b['l1'], dtype=torch.long)
    _x2 = torch.tensor(b['x2'], dtype=torch.long)
    _l2 = torch.tensor(b['l2'], dtype=torch.long)
    _y = torch.tensor(b['y'], dtype=torch.long)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    
    o = model(_x1, _x2, _l1, _l2)
    
    loss = torch.nn.CrossEntropyLoss()
    _c = loss(o, _y)
    
    _c.backward()
    optimizer.step()
    
