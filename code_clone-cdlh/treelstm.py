# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:32:43 2021

@author: DrLC
"""

import torch
import torch as th
import torch.nn as nn
import dgl
from dgl.nn import GlobalAttentionPooling, MaxPooling, AvgPooling

class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}

class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 n_class,
                 dropout,
                 device=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.vocab_size = num_vocabs
        self.embedding = nn.Embedding(num_vocabs, x_size)
        self.dropout = nn.Dropout(dropout)
        #self.pool = GlobalAttentionPooling(nn.Linear(h_size,1))
        #self.pool = MaxPooling()
        self.pool = AvgPooling()
        #self.pool = nn.AdaptiveMaxPool1d(1)
        self.classify = nn.Linear(h_size, n_class)
        cell = ChildSumTreeLSTMCell
        self.cell = cell(x_size, h_size)
        self.device = device

    def _forward(self, g):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        g : dgl.DGLGraph
            Tree for computation.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        embeds = self.embedding(g.ndata['type'])
        n = g.number_of_nodes()
        h = th.zeros((n, self.h_size)).to(self.device)
        c = th.zeros((n, self.h_size)).to(self.device)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds))
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # compute logits
        #h = self.dropout(g.ndata.pop('h'))
        #rep = self.pool(h.transpose(0, 1 ).unsqueeze(1)).squeeze().unsqueeze(0)
        rep = self.pool(g, g.ndata['h'])
        return rep
    
    def forward(self, g1, g2):
        
        rep1 = self._forward(g1)
        rep2 = self._forward(g2)
        abs_dist = th.abs(th.add(rep1, -rep2))
        logits = self.classify(abs_dist)
        return logits

    def prob(self, g1, g2):
        logits = self.forward(g1, g2)
        prob = nn.Softmax(dim=-1)(logits)
        return prob

    def grad(self, g1, g2, labels, loss_fn):

        rep2 = self._forward(g2)
        rep2_const = rep2.clone().detach() #torch.tensor(rep2, requires_grad=False)

        self.zero_grad()
        rep1 = self._forward(g1)
        abs_dist = torch.abs(torch.add(rep1, -rep2_const))
        logits = self.classify(abs_dist)
        #prob = nn.Softmax(dim=-1)(logits)

        self.embedding.weight.retain_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        #print(self.embedding.weight.grad.size())

        return self.embedding.weight.grad
