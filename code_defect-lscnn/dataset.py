# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:18:29 2021

@author: DrLC
"""

import pickle, gzip
import random
import numpy
import copy

from utils import brute_stmt_split

class Dataset(object):
    
    def __init__(self, xs=[], ys=[], raws=None, ids=None, idx2txt=[], txt2idx={},
                 max_stmt_cnt=40, max_stmt_len=20, vocab_size=3000, dtype=None):
        
        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2txt = idx2txt
        self.__txt2idx = txt2idx
        assert len(self.__idx2txt) == self.__vocab_size \
            and len(self.__txt2idx) == self.__vocab_size + 1
        self.__max_stmt_len = max_stmt_len
        self.__max_stmt_cnt = max_stmt_cnt
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
            stmts = brute_stmt_split(x, self.__idx2txt, keep_raw=True)
            self.__xs.append([])
            self.__ls.append([])
            for s in stmts[: self.__max_stmt_cnt]:
                if len(s) <= 0:
                    continue
                self.__xs[-1].append([])
                for t in s[: self.__max_stmt_len]:
                    if t >= self.__vocab_size:
                        self.__xs[-1][-1].append(self.__txt2idx['<unk>'])
                    else:
                        self.__xs[-1][-1].append(t)
                if len(s) < self.__max_stmt_len:
                    self.__xs[-1][-1] += [self.__txt2idx['<pad>']] * (self.__max_stmt_len - len(s))
                    self.__ls[-1].append(len(s))
                else:
                    self.__ls[-1].append(self.__max_stmt_len)
            if len(self.__xs[-1]) < self.__max_stmt_cnt:
                self.__ls[-1] += [-1] * (self.__max_stmt_cnt - len(self.__xs[-1]))
                self.__xs[-1] += [[self.__txt2idx['<pad>']] * self.__max_stmt_len] \
                            * (self.__max_stmt_cnt - len(self.__xs[-1]))
            self.__xs[-1] = numpy.asarray(self.__xs[-1])
        self.__xs = numpy.asarray(self.__xs, dtype=self.__dtype['int'])
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ls = numpy.asarray(self.__ls, dtype=self.__dtype['int'])
        assert 0 not in self.__ls
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
    
class CODECHEF(object):
    
    def __init__(self, path='../data_defect/codechef.pkl.gz', max_stmt_len=20, max_stmt_cnt=40,
                 vocab_size=3000, valid_ratio=0.2, dtype='32'):
        
        self.__dtypes = self.__dtype(dtype)
        self.__max_stmt_len = max_stmt_len
        self.__max_stmt_cnt = max_stmt_cnt
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
                           max_stmt_len=self.__max_stmt_len,
                           max_stmt_cnt=self.__max_stmt_cnt,
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
                             max_stmt_len=self.__max_stmt_len,
                             max_stmt_cnt=self.__max_stmt_cnt,
                             vocab_size=self.__vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(xs=d['x_te'],
                            ys=d['y_te'],
                            raws=d['raw_te'],
                            idx2txt=self.__idx2txt,
                            txt2idx=self.__txt2idx,
                            max_stmt_len=self.__max_stmt_len,
                            max_stmt_cnt=self.__max_stmt_cnt,
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
        
    def get_max_stmt_cnt(self):

        return self.__max_stmt_cnt

    def get_max_stmt_len(self):

        return self.__max_stmt_len

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
            
    def idxs2stmts(self, xs, ls):
        
        stmts = []
        for x, l in zip(xs, ls):
            stmts.append([])
            for _s, _l in zip(x, l):
                stmts[-1].append([])
                if _l < 0:
                    break
                for t in _s[: _l]:
                    stmts[-1][-1].append(self.__idx2txt[t])
        return stmts
            
if __name__ == "__main__":
    
    import time
    start_time = time.time()
    oj = CODECHEF(path="../data_defect/codechef.pkl.gz")
    print ("time cost = "+str(time.time()-start_time)+" sec")

    print ("==========")
    start_time = time.time()
    b = oj.train.next_batch(10)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for stmt in brute_stmt_split(b['raw'][0]):
        print (" ".join(stmt))
    print ()
    for stmt in oj.idxs2stmts(b['x'][:1], b['l'][:1])[0]:
        print (" ".join(stmt))
    print ()
    
    print ("==========")
    start_time = time.time()
    b = oj.dev.next_batch(10)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for stmt in brute_stmt_split(b['raw'][0]):
        print (" ".join(stmt))
    print ()
    for stmt in oj.idxs2stmts(b['x'][:1], b['l'][:1])[0]:
        print (" ".join(stmt))
    print ()
    
    print ("==========")
    start_time = time.time()
    b = oj.test.next_batch(10)
    print ("time cost = "+str(time.time()-start_time)+" sec")
    for stmt in brute_stmt_split(b['raw'][0]):
        print (" ".join(stmt))
    print ()
    for stmt in oj.idxs2stmts(b['x'][:1], b['l'][:1])[0]:
        print (" ".join(stmt))
    print ()
