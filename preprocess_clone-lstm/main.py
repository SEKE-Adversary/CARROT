# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:05:36 2020

@author: DrLC
"""

import pickle, gzip
import build_clone_dataset as bcd
import random
import tqdm

def build(data_path="../data/oj.pkl.gz",
          symtab_path = '../data/oj_uid.pkl.gz',
          stmt_inspos_path='../data/oj_inspos.pkl.gz',
          subset=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
          n_train=40000, n_test=10000):
    
    with gzip.open(data_path, "rb") as f:
        d = pickle.load(f)
    with gzip.open(symtab_path, "rb") as f:
        sym = pickle.load(f)
    with gzip.open(stmt_inspos_path, "rb") as f:
        ins = pickle.load(f)
        
    train_raw, train_y, train_uid, train_stmt, train_decl = [], [], [], [], []
    for i in range(len(d["raw_tr"])):
        if d['y_tr'][i] in subset:
            train_raw.append(d['raw_tr'][i])
            train_y.append(d['y_tr'][i])
            train_uid.append(sym['tr'][i])
            train_stmt.append(ins['stmt_tr'][i])
            train_decl.append(ins['decl_tr'][i])
    test_raw, test_y, test_uid, test_stmt, test_decl = [], [], [], [], []
    for i in range(len(d["raw_te"])):
        if d['y_te'][i] in subset:
            test_raw.append(d['raw_te'][i])
            test_y.append(d['y_te'][i])
            test_uid.append(sym['te'][i])
            test_stmt.append(ins['stmt_te'][i])
            test_decl.append(ins['decl_te'][i])
    
    all_uid = []
    for _uids in train_uid:
        for _uid in _uids.keys():
            if _uid not in all_uid:
                all_uid.append(_uid)
    idx2txt, txt2idx = bcd.build_vocab(train_raw)
    train_x = bcd.text2index(train_raw, txt2idx)
    test_x = bcd.text2index(test_raw, txt2idx)
    
    train_pair, train_label = [], []
    for _ in tqdm.tqdm(range(n_train)):
        rand1, rand2 = random.sample(range(len(train_y)), 2)
        while (rand1, rand2) in train_pair:
            rand1, rand2 = random.sample(range(len(train_y)), 2)
        train_pair.append((rand1, rand2))
        if train_y[rand1] == train_y[rand2]:
            train_label.append(1)
        else:
            train_label.append(0) 
    test_pair, test_label = [], []
    for _ in tqdm.tqdm(range(n_test)):
        rand1, rand2 = random.sample(range(len(test_y)), 2)
        while (rand1, rand2) in test_pair:
            rand1, rand2 = random.sample(range(len(test_y)), 2)
        test_pair.append((rand1, rand2))
        if test_y[rand1] == test_y[rand2]:
            test_label.append(1)
        else:
            test_label.append(0)
            
    d = {"tr": {"pair": train_pair, "label": train_label,
                "raw": train_raw, "y": train_y, "x": train_x},
         "te": {"pair": test_pair, "label": test_label,
                "raw": test_raw, "y": test_y, "x": test_x},
         "idx2txt": idx2txt, "txt2idx": txt2idx}
    sym = {"tr": train_uid, "te": test_uid, "all": all_uid}
    ins = {"stmt_tr": train_stmt, "stmt_te": test_stmt, 
           "decl_tr": train_decl, "decl_te": test_decl}
    return d, sym, ins

if __name__ == "__main__":
    
    random.seed(1726)
    
    data_path="../data_clone/oj.pkl.gz"
    symtab_path = '../data_clone/oj_uid.pkl.gz'
    stmt_inspos_path='../data_clone/oj_inspos.pkl.gz'
    d, sym, ins = build()
    with gzip.open(data_path, "wb") as f:
        pickle.dump(d, f)
    with gzip.open(symtab_path, "wb") as f:
        pickle.dump(sym, f)
    with gzip.open(stmt_inspos_path, "wb") as f:
        pickle.dump(ins, f)