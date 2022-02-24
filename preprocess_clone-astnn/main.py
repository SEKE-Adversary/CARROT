# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:54:47 2020

@author: DrLC
"""

import os
import gzip
import pickle
import pandas as pd
import random
from copy import deepcopy as cp

from transform import dump_to_pkl, check_or_create
from pipeline import Pipeline

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
    
    train_path = '../data_clone/train/'
    check_or_create(train_path)
    dev_path = '../data_clone/dev/'
    check_or_create(dev_path)
    test_path = '../data_clone/test/'
    check_or_create(test_path)
    
    dump_to_pkl(train, '../data_clone/train/program.pkl', '../data_clone/train/pair.pkl', base=0)
    dump_to_pkl(dev, '../data_clone/dev/program.pkl', '../data_clone/dev/pair.pkl', base=0)
    # 防止train, dev, test的program id有重复
    test_base = len(train['pair']) + len(dev['pair'])
    dump_to_pkl(test, '../data_clone/test/program.pkl', '../data_clone/test/pair.pkl', base=test_base)
    
    ppl = Pipeline('../data_clone/')
    ppl.run()