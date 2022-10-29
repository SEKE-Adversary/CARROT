# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:57:37 2021

@author: DrLC
"""

import pickle, gzip
import random
import numpy
import copy

import dgl
import torch

class Dataset(object):
    
    def __init__(self, nodes, begins, ends, xs, ys, raws, ids=None,
                 idx2node=[], node2idx={}, vocab_size=2000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2node = idx2node
        self.__node2idx = node2idx
        self.__xs = numpy.asarray(xs, dtype=self.__dtype['int'])
        self.__ys = numpy.asarray(ys, dtype=self.__dtype['int'])
        self.__raws = {}
        self.__graphs = {}
        if ids is None:
            ids = list(range(len(nodes)))
            
        assert len(ids) == len(nodes) \
            and len(nodes) == len(begins) \
            and len(begins) == len(ends)
        
        for n, b, e, i, j in zip(nodes, begins, ends, ids, range(len(ids))):
            self.__raws[i] = copy.deepcopy(raws[j])
            self.__graphs[i] = dgl.graph((torch.tensor(b), torch.tensor(e)))
            for ii in range(len(n)):
                if n[ii] >= vocab_size:
                    n[ii] = self.__node2idx["<unk>"]
            self.__graphs[i].ndata['type'] = torch.tensor(n)
        self.__size = len(self.__xs)
        
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size=32):
        
        batch = {"graph1": [], "raw1": [], "id1": [], "offset1": [],
                 "graph2": [], "raw2": [], "id2": [], "offset2": [],
                 "y": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['graph1'] = dgl.batch([self.__graphs[self.__xs[i][0]] for i in idxs])
        batch['graph2'] = dgl.batch([self.__graphs[self.__xs[i][1]] for i in idxs])
        batch['raw1'] = copy.deepcopy([self.__raws[self.__xs[i][0]] for i in idxs])
        batch['raw2'] = copy.deepcopy([self.__raws[self.__xs[i][1]] for i in idxs])
        batch['id1'] = numpy.take(self.__xs, indices=idxs, axis=0)[:, 0]
        batch['id2'] = numpy.take(self.__xs, indices=idxs, axis=0)[:, 1]
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        return batch
        
    def get_size(self):
        
        return self.__size
        
    def get_rest_epoch_size(self):
        
        return len(self.__epoch)

class OJ_AST_GRAPH(object):
    
    def __init__(self, path='../data_clone/oj_ast_graph.pkl.gz', vocab_size=2000,
                 valid_ratio=0.2, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        self.__vocab_size = vocab_size
        
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        
        self.__idx2node = d['idx2node'][:self.__vocab_size]
        self.__node2idx = {"<pad>": 0}
        for i, t in zip(range(vocab_size), self.__idx2node):
            self.__node2idx[t] = i
            assert self.__node2idx[t] == d['node2idx'][t]
            
        idxs = random.sample(range(len(d['tr']['pair'])), len(d['tr']['pair']))
        n_valid = int(len(d['tr']['pair']) * valid_ratio)
        xs, ys = [], []
        for i in idxs[:n_valid]:
            xs.append(d['tr']['pair'][i])
            ys.append(d['tr']['label'][i])
        self.dev = Dataset(nodes=d['tr']['node'],
                           begins=d['tr']['begin'],
                           ends=d['tr']['end'],
                           xs=xs,
                           ys=ys,
                           raws=d['tr']['raw'],
                           idx2node=self.__idx2node,
                           node2idx=self.__node2idx,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        xs, ys = [], []
        for i in idxs[n_valid:]:
            xs.append(d['tr']['pair'][i])
            ys.append(d['tr']['label'][i])
        self.train = Dataset(nodes=d['tr']['node'],
                             begins=d['tr']['begin'],
                             ends=d['tr']['end'],
                             xs=xs,
                             ys=ys,
                             raws=d['tr']['raw'],
                             idx2node=self.__idx2node,
                             node2idx=self.__node2idx,
                             vocab_size=self.__vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(nodes=d['te']['node'],
                            begins=d['te']['begin'],
                            ends=d['te']['end'],
                            xs=d['te']['pair'],
                            ys=d['te']['label'],
                            raws=d['te']['raw'],
                            idx2node=self.__idx2node,
                            node2idx=self.__node2idx,
                            vocab_size=self.__vocab_size,
                            dtype=self.__dtypes)

    def __dtype(self, dtype='32'):
    
        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):
        
        return self.__dtypes
    
    def get_max_node(self):
        
        return self.__max_node
        
    def get_vocab_size(self):
        
        return self.__vocab_size
        
    def get_idx2node(self):
        
        return copy.deepcopy(self.__idx2node)
    
    def get_node2idx(self):
        
        return copy.deepcopy(self.__node2idx)
        
    def node2idx(self, node):
        
        if node in self.__node2idx.keys():
            return self.__node2idx[node]
        else:
            return self.__node2idx['<unk>']

    def idx2node(self, idx):
        
        if idx >= 0 and idx < len(self.__idx2node):
            return self.__idx2node[idx]
        else:
            return '<unk>'
            
    def vis_graph(self, g):
        
        try:
            import networkx as nx
            nx_g = g.to_networkx()
            labels = {}
            for i in range(len(g.ndata['type'])):
                if g.ndata['type'][i] > 0 and g.ndata['type'][i] < len(self.__idx2node):
                    labels[i] = self.__idx2node[g.ndata['type'][i]]
                else:
                    labels[i] = "<unk>"
            pos = nx.drawing.nx_pydot.graphviz_layout(nx_g, prog="dot")
            nx.draw(nx_g, pos, labels=labels, with_labels=True, node_color=['#ffffff' for _ in labels])
        except  ImportError:
            pass
            
if __name__ == "__main__":
    
    import time
    from treelstm import TreeLSTM
    
    start_time = time.time()
    oj = OJ_AST_GRAPH(path="../data_clone/oj_ast_graph.pkl.gz")
    print ("time cost = "+str(time.time()-start_time)+" sec")

    start_time = time.time()
    b = oj.train.next_batch(4)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    
    model = TreeLSTM(num_vocabs=oj.get_vocab_size(),
                     x_size=101,
                     h_size=103,
                     n_class=2,
                     dropout=0.5,)
    logits = model(b['graph1'], b['graph2'])