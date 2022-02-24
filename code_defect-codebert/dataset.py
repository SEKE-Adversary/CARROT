# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:00:17 2021

@author: DrLC
"""

import pickle, gzip
import random
import numpy
import copy

class Dataset(object):
    
    def __init__(self, xs=[], ys=[], raws=None, ids=None, idx2txt=[], txt2idx={},
                 vocab_size=5000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2txt = idx2txt
        self.__txt2idx = txt2idx
        assert len(self.__idx2txt) == self.__vocab_size \
            and len(self.__txt2idx) == self.__vocab_size + 1
        self.__xs = []
        self.__raws = []
        self.__ys = []
        self.__ids = []
        if raws is None:
            assert len(xs) == len(ys)
            raws = [None for _ in ys]
        else:
            assert len(xs) == len(ys) and len(ys) == len(raws)
        if ids is None:
            ids = list(range(len(xs)))
        else:
            assert len(xs) == len(ids)
        for x, y, r, i in zip(xs, ys, raws, ids):
            self.__raws.append(r)
            self.__ys.append(y)
            self.__ids.append(i)
            self.__xs.append([])
            for t in x:
                if t >= self.__vocab_size:
                    self.__xs[-1].append('<unk>')
                else:
                    self.__xs[-1].append(self.__idx2txt[t])
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ids = numpy.asarray(self.__ids, dtype=self.__dtype['int'])
        self.__size = len(self.__raws)
        
        assert self.__size == len(self.__raws)      \
            and len(self.__raws) == len(self.__xs)  \
            and len(self.__xs) == len(self.__ys) \
            and len(self.__ys) == len(self.__ids)
        
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size=32):
        
        batch = {"x": [], "y": [], "raw": [], "id": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['x'] = copy.deepcopy([" ".join(self.__xs[i]) for i in idxs])
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        batch['id'] = numpy.take(self.__ids, indices=idxs, axis=0)
        batch['raw'] = copy.deepcopy([self.__raws[i] for i in idxs])
        return batch
        
    def get_size(self):
        
        return self.__size
        
    def get_rest_epoch_size(self):
        
        return len(self.__epoch)

class CODECHEF(object):
    
    def __init__(self, path='../data_defect/codechef.pkl.gz', vocab_size=-1,
                 valid_ratio=0.2, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        
        if vocab_size > 0:
            self.__idx2txt = d['idx2txt'][:self.__vocab_size]
            self.__vocab_size = vocab_size
        else:
            self.__idx2txt = d['idx2txt']
            self.__vocab_size = len(self.__idx2txt)
        self.__txt2idx = {"<pad>": 0}
        for i, t in zip(range(self.__vocab_size), self.__idx2txt):
            self.__txt2idx[t] = i
            
        idxs = random.sample(range(len(d['x_tr'])), len(d['x_tr']))
        n_valid = int(len(d['x_tr'])*valid_ratio)
        raw, x, y, ids = [], [], [], []
        for i in idxs[:n_valid]:
            raw.append(d['raw_tr'][i])
            x.append(d['x_tr'][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        self.dev = Dataset(xs=x, ys=y, raws=raw, ids=ids,
                           idx2txt=self.__idx2txt,
                           txt2idx=self.__txt2idx,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        raw, x, y, ids = [], [], [], []
        for i in idxs[n_valid:]:
            raw.append(d['raw_tr'][i])
            x.append(d['x_tr'][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        self.train = Dataset(xs=x, ys=y, raws=raw, ids=ids,
                           idx2txt=self.__idx2txt,
                           txt2idx=self.__txt2idx,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        self.test = Dataset(xs=d['x_te'],
                            ys=d['y_te'],
                            raws=d['raw_te'],
                            idx2txt=self.__idx2txt,
                            txt2idx=self.__txt2idx,
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
            
if __name__ == "__main__":
    
    import time
    start_time = time.time()
    cc = CODECHEF(path="../data/oj.pkl.gz")
    print ("time cost = "+str(time.time()-start_time)+" sec")


    start_time = time.time()
    b = cc.train.next_batch(2)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for x in b['x']:
        print (x)
    