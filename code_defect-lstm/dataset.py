# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:32:53 2020

@author: DrLC
"""

import pickle, gzip
import random
import numpy
import copy

class Dataset(object):
    
    def __init__(self, xs=[], ys=[], raws=None, ids=None, idx2txt=[], txt2idx={},
                 max_len=300, vocab_size=3000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2txt = idx2txt
        self.__txt2idx = txt2idx
        self.__max_len = max_len
        self.__xs = []
        self.__raws = []
        self.__ys = []
        self.__ls = []
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
            if len(x) > self.__max_len:
                self.__ls.append(self.__max_len)
            else:
                self.__ls.append(len(x))
            self.__xs.append([])
            for t in x[:self.__max_len]:
                if t >= self.__vocab_size:
                    self.__xs[-1].append(self.__txt2idx['<unk>'])
                else:
                    self.__xs[-1].append(t)
            while len(self.__xs[-1]) < self.__max_len:
                self.__xs[-1].append(self.__txt2idx['<pad>'])
        self.__xs = numpy.asarray(self.__xs, dtype=self.__dtype['int'])
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ls = numpy.asarray(self.__ls, dtype=self.__dtype['int'])
        self.__ids = numpy.asarray(self.__ids, dtype=self.__dtype['int'])
        self.__size = len(self.__raws)
        
        assert self.__size == len(self.__raws)      \
            and len(self.__raws) == len(self.__xs)  \
            and len(self.__xs) == len(self.__ys) \
            and len(self.__ys) == len(self.__ls) \
            and len(self.__ls) == len(self.__ids)
        
        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):
        
        self.__epoch = random.sample(range(self.__size), self.__size)
        
    def next_batch(self, batch_size=32):
        
        batch = {"x": [], "y": [], "l": [], "raw": [], "id": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['x'] = numpy.take(self.__xs, indices=idxs, axis=0)
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        batch['l'] = numpy.take(self.__ls, indices=idxs, axis=0)
        batch['id'] = numpy.take(self.__ids, indices=idxs, axis=0)
        for i in idxs:
            batch['raw'].append(self.__raws[i])
        batch['raw'] = copy.deepcopy(batch['raw'])
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
        
class CodeChef(object):
    
    def __init__(self, path='../data_defect/codechef.pkl.gz', max_len=300, vocab_size=3000,
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
                           max_len=self.__max_len,
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
                           max_len=self.__max_len,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        self.test = Dataset(xs=d['x_te'],
                           ys=d['y_te'],
                           raws=d['raw_te'],
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
    cc = CodeChef()
    print ("time cost = "+str(time.time()-start_time)+" sec")

    start_time = time.time()
    b = cc.train.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw'][0]:
        print (t, end=" ")
    print ()
    for t in cc.idxs2raw(b['x'], b['l'])[0]:
        print (t, end=" ")
    print ("\n")
    start_time = time.time()
    b = cc.dev.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw'][0]:
        print (t, end=" ")
    print ()
    for t in cc.idxs2raw(b['x'], b['l'])[0]:
        print (t, end=" ")
    print ("\n")
    start_time = time.time()
    b = cc.test.next_batch(1)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for t in b['raw'][0]:
        print (t, end=" ")
    print ()
    for t in cc.idxs2raw(b['x'], b['l'])[0]:
        print (t, end=" ")
    print ("\n")
