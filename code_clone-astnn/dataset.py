# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:07:01 2020

@author: DrLC
"""

import pickle, gzip
import random
import numpy
import copy

class Dataset(object):
    
    def __init__(self, xs=[], ys=[], seqs=[], raws=None, ids=None, classes=None,
                 idx2txt=[], txt2idx={}, max_len=300, vocab_size=5000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2txt = idx2txt
        self.__txt2idx = txt2idx
        assert len(self.__idx2txt) == self.__vocab_size \
            and len(self.__txt2idx) == self.__vocab_size + 1
        self.__max_len = max_len
        self.__seqs = {}
        self.__raws = {}
        self.__ls = {}
        self.__classes = {}
        if classes is None:
            classes = [None for _ in seqs]
        if raws is None:
            raws = [None for _ in seqs]
        assert len(xs) == len(ys)
        if ids is None:
            ids = list(range(len(seqs)))
        for s, r, i, c in zip(seqs, raws, ids, classes):
            self.__raws[i] = r
            self.__classes[i] = c
            if len(s) > self.__max_len:
                self.__ls[i] = self.__max_len
            else:
                self.__ls[i] = len(s)
            self.__seqs[i] = []
            for t in s[:self.__max_len]:
                if t >= self.__vocab_size:
                    self.__seqs[i].append(self.__txt2idx['<unk>'])
                else:
                    self.__seqs[i].append(t)
            while len(self.__seqs[i]) < self.__max_len:
                self.__seqs[i].append(self.__txt2idx['<pad>'])
        self.__xs = numpy.asarray(xs, dtype=self.__dtype['int'])
        self.__ys = numpy.asarray(ys, dtype=self.__dtype['int'])
        self.__size = len(self.__xs)
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size=32):
        
        batch = {"x1": [], "l1": [], "raw1": [], "y1": [], "id1": [],
                 "x2": [], "l2": [], "raw2": [], "y2": [], "id2": [],
                 "y": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        idx_pairs = numpy.take(self.__xs, indices=idxs, axis=0)
        for idx1, idx2 in idx_pairs:
            batch['x1'].append(self.__seqs[idx1])
            batch['l1'].append(self.__ls[idx1])
            batch['raw1'].append(self.__raws[idx1])
            batch['y1'].append(self.__classes[idx1])
            batch['id1'].append(idx1)
            batch['x2'].append(self.__seqs[idx2])
            batch['l2'].append(self.__ls[idx2])
            batch['raw2'].append(self.__raws[idx2])
            batch['y2'].append(self.__classes[idx2])
            batch['id2'].append(idx2)
        batch['x1'] = numpy.asarray(batch['x1'])
        batch['x2'] = numpy.asarray(batch['x2'])
        batch['l1'] = numpy.asarray(batch['l1'])
        batch['l2'] = numpy.asarray(batch['l2'])
        batch['y1'] = numpy.asarray(batch['y1'])
        batch['y2'] = numpy.asarray(batch['y2'])
        batch['id1'] = numpy.asarray(batch['id1'])
        batch['id2'] = numpy.asarray(batch['id2'])
        batch['raw1'] = copy.deepcopy(batch['raw1'])
        batch['raw2'] = copy.deepcopy(batch['raw2'])
        return batch
        
    def idxs2raw(self, xs, ls):
        
        seq = []
        for x, l in zip(xs, ls):
            seq.append([])
            for t in x[:l]:
                seq[-1].append(self.__idx2txt[t])
        return seq
        
    def get_size(self):
        
        return self.__size
        
    def get_rest_epoch_size(self):
        
        return len(self.__epoch)
        
class OJ104(object):
    
    def __init__(self, path='../data_clone/oj.pkl.gz', max_len=500, vocab_size=2000,
                 valid_ratio=0.2, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        self.__max_len = max_len
        self.__vocab_size = vocab_size
        
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        
        self.__idx2txt = d['idx2txt'][:self.__vocab_size]
        self.__txt2idx = {"<pad>": 0}
        for i, t in zip(range(vocab_size), self.__idx2txt):
            self.__txt2idx[t] = i
            assert self.__txt2idx[t] == d['txt2idx'][t]
            
        idxs = random.sample(range(len(d['tr']['pair'])), len(d['tr']['pair']))
        n_valid = int(len(d['tr']['pair'])*valid_ratio)
        x, y= [], []
        for i in idxs[:n_valid]:
            x.append(d['tr']['pair'][i])
            y.append(d['tr']['label'][i])
        self.dev = Dataset(xs=x, ys=y,
                           seqs=d['tr']['x'],
                           raws=d['tr']['raw'],
                           classes=d['tr']['y'],
                           idx2txt=self.__idx2txt,
                           txt2idx=self.__txt2idx,
                           max_len=self.__max_len,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        x, y = [], []
        for i in idxs[n_valid:]:
            x.append(d['tr']['pair'][i])
            y.append(d['tr']['label'][i])
        self.train = Dataset(xs=x, ys=y,
                             seqs=d['tr']['x'],
                             raws=d['tr']['raw'],
                             classes=d['tr']['y'],
                             idx2txt=self.__idx2txt,
                             txt2idx=self.__txt2idx,
                             max_len=self.__max_len,
                             vocab_size=self.__vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(xs=d['te']['pair'],
                            ys=d['te']['label'],
                            seqs=d['te']['x'],
                            raws=d['te']['raw'],
                            classes=d['te']['y'],
                            idx2txt=self.__idx2txt,
                            txt2idx=self.__txt2idx,
                            max_len=self.__max_len,
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
    
    def get_max_len(self):
        
        return self.__max_len
        
    def get_vocab_size(self):
        
        return self.__vocab_size
        
    def get_idx2txt(self):
        
        return copy.deepcopy(self.__idx2txt)
    
    def get_txt2idx(self):
        
        return copy.deepcopy(self.__txt2idx)
        
    def vocab2idx(self, vocab):
        
        if vocab in self.__txt2idx.keys():
            return self.__txt2idx[vocab]
        else:
            return self.__txt2idx['<unk>']

    def idx2vocab(self, idx):
        
        if idx >= 0 and idx < len(self.__idx2txt):
            return self.__idx2txt[idx]
        else:
            return '<unk>'
            
    def idxs2raw(self, xs, ls):
        
        seq = []
        for x, l in zip(xs, ls):
            seq.append([])
            for t in x[:l]:
                seq[-1].append(self.__idx2txt[t])
        return seq
            
if __name__ == "__main__":
    
    import time
    start_time = time.time()
    oj = OJ104(path="../data_clone/oj.pkl.gz")
    print ("time cost = "+str(time.time()-start_time)+" sec")

    start_time = time.time()
    b = oj.train.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw1'][0]:
        print (t, end=" ")
    print ()
    for t in oj.idxs2raw(b['x1'], b['l1'])[0]:
        print (t, end=" ")
    print ("\n")
    for t in b['raw2'][0]:
        print (t, end=" ")
    print ()
    for t in oj.idxs2raw(b['x2'], b['l2'])[0]:
        print (t, end=" ")
    print ("\n")
    
    start_time = time.time()
    b = oj.dev.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw1'][0]:
        print (t, end=" ")
    print ()
    for t in oj.idxs2raw(b['x1'], b['l1'])[0]:
        print (t, end=" ")
    print ("\n")
    for t in b['raw2'][0]:
        print (t, end=" ")
    print ()
    for t in oj.idxs2raw(b['x2'], b['l2'])[0]:
        print (t, end=" ")
    print ("\n")
    
    start_time = time.time()
    b = oj.test.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw1'][0]:
        print (t, end=" ")
    print ()
    for t in oj.idxs2raw(b['x1'], b['l1'])[0]:
        print (t, end=" ")
    print ("\n")
    for t in b['raw2'][0]:
        print (t, end=" ")
    print ()
    for t in oj.idxs2raw(b['x2'], b['l2'])[0]:
        print (t, end=" ")
    print ("\n")