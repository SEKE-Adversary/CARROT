# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:31:25 2020

@author: DrLC
"""

import os
import gzip
import pickle
import pandas as pd
import random
from copy import deepcopy as cp

# convert dict data to programs and pairs csv data
def dump_to_pkl(data, program_path, pair_path, base=0):
    size = len(data['raw'])
    dic = dict()
    dic['id'] = [i + base for i in list(range(size))]
    dic['code'] = [' '.join(data['raw'][i]) for i in range(size)]
    dic['label'] = [data['label'][i] for i in range(size)]
    frame = pd.DataFrame.from_dict(dic)
    with open(program_path, 'wb') as f:
        pickle.dump(frame, f)
    dic2 = dict()
    dic2['id1'] = [p[0] + base for p in data['pair']]
    dic2['id2'] = [p[1] + base for p in data['pair']]
    dic2['label'] = data['label']
    frame2 = pd.DataFrame.from_dict(dic2)
    with open(pair_path, 'wb') as f:
        pickle.dump(frame2, f)
    return

def check_or_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
if __name__ == "__main__":
    
    data_path = '../data_clone/oj.pkl.gz'
    split_ratio = 0.75
    
    f = gzip.open(data_path, 'rb')
    poj104 = pickle.load(f)
    
    train = poj104['tr']
    test = poj104['te']
    assert(len(train['raw']) == len(train['y']))
    
    
    # split train, dev data
    dev = cp(train)
    train_dev_size = len(train['pair'])
    idxs = random.sample(range(train_dev_size), train_dev_size)
    split = int(len(train['pair']) * split_ratio)
    train['pair'] = [train['pair'][idx] for idx in idxs[:split]]
    train['label'] = [train['label'][idx] for idx in idxs[:split]]
    dev['pair'] = [dev['pair'][idx] for idx in idxs[split:]]
    dev['label'] = [dev['label'][idx] for idx in idxs[split:]]
        
    # 防止train, dev, test的program id有重复
    test_base = len(train['pair']) + len(dev['pair'])
        
    train_path = '../data_clone/train/'
    check_or_create(train_path)
    dev_path = '../data_clone/dev/'
    check_or_create(dev_path)
    test_path = '../data_clone/test/'
    check_or_create(test_path)
    
    dump_to_pkl(train, '../data_clone/train/program.pkl', '../data_clone/train/pair.pkl', base=0)
    dump_to_pkl(dev, '../data_clone/dev/program.pkl', '../data_clone/dev/pair.pkl', base=0)
    dump_to_pkl(test, '../data_clone/test/program.pkl', '../data_clone/test/pair.pkl', base=test_base)
