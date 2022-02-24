# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:49:38 2021

@author: DrLC
"""

import pickle, gzip
import random
import numpy
import copy

import dgl
import torch

class Dataset(object):
    
    def __init__(self, nodes, begins, ends, ys, raws=None, ids=None,
                 idx2node=[], node2idx={}, vocab_size=5000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2node = idx2node
        self.__node2idx = node2idx
        self.__graphs = []
        self.__raws = []
        self.__ys = []
        self.__ids = []
        if raws is None:
            raws = [None for _ in ys]
        if ids is None:
            ids = list(range(len(nodes)))
        for n, b, e, y, r, i in zip(nodes, begins, ends, ys, raws, ids):
            self.__raws.append(r)
            self.__ys.append(y)
            self.__ids.append(i)
            self.__graphs.append(dgl.graph((torch.tensor(b), torch.tensor(e))))
            for ii in range(len(n)):
                if n[ii] >= vocab_size:
                    n[ii] = self.__node2idx["<unk>"]
            self.__graphs[-1].ndata['type'] = torch.tensor(n)
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ids = numpy.asarray(self.__ids, dtype=self.__dtype['int'])
        self.__size = len(self.__raws)
        
        assert self.__size == len(self.__raws)      \
            and len(self.__raws) == len(self.__graphs)  \
            and len(self.__graphs) == len(self.__ys) \
            and len(self.__ys) == len(self.__ids)
        
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size=32):
        
        batch = {"graph": [], "y": [], "raw": [], "id": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['graph'] = dgl.batch([self.__graphs[i] for i in idxs])
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        batch['id'] = numpy.take(self.__ids, indices=idxs, axis=0)
        batch['raw'] = [self.__raws[i] for i in idxs]
        batch['raw'] = copy.deepcopy(batch['raw'])
        return batch
        
    def get_size(self):
        
        return self.__size
        
    def get_rest_epoch_size(self):
        
        return len(self.__epoch)

class OJ104_AST_GRAPH(object):
    
    def __init__(self, path='../data/oj_ast_graph.pkl.gz', vocab_size=5000,
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
            
        idxs = random.sample(range(len(d['node_tr'])), len(d['node_tr']))
        n_valid = int(len(d['node_tr']) * valid_ratio)
        raw, node, begin, end, y, ids = [], [], [], [], [], []
        for i in idxs[:n_valid]:
            raw.append(d['raw_tr'][i])
            node.append(d["node_tr"][i])
            begin.append(d["begin_tr"][i])
            end.append(d["end_tr"][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        self.dev = Dataset(nodes=node, begins=begin, ends=end,
                           ys=y,raws=raw, ids=ids,
                           idx2node=self.__idx2node,
                           node2idx=self.__node2idx,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        raw, node, begin, end, y, ids = [], [], [], [], [], []
        for i in idxs[n_valid:]:
            raw.append(d['raw_tr'][i])
            node.append(d["node_tr"][i])
            begin.append(d["begin_tr"][i])
            end.append(d["end_tr"][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        self.train = Dataset(nodes=node, begins=begin, ends=end,
                             ys=y, raws=raw, ids=ids,
                             idx2node=self.__idx2node,
                             node2idx=self.__node2idx,
                             vocab_size=self.__vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(nodes=d['node_te'],
                            begins=d['begin_te'],
                            ends=d['end_te'],
                            ys=d['y_te'],
                            raws=d['raw_te'],
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
    start_time = time.time()
    oj = OJ104_AST_GRAPH(path="../data/oj_ast_graph.pkl.gz")
    print ("time cost = "+str(time.time()-start_time)+" sec")

    start_time = time.time()
    b = oj.train.next_batch(2)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    
    from tbcnn import TBCNNClassifier
    
    model = TBCNNClassifier(x_size=300,
                            h_size=256,
                            dropout=0.5,
                            n_classes=104,
                            vocab_size=5010)
    model.forward(b['graph'])
