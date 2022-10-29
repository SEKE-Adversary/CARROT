# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:14:16 2021

@author: DrLC
"""

def brute_stmt_split(_tokens, idx2txt=None, keep_raw=True):
    
    split_pos = []
    skip_cnt = 0
    if idx2txt:
        tokens = [idx2txt[t] if t < len(idx2txt) else idx2txt[0]
                  for t in _tokens]
    else:
        tokens = _tokens
    for i, t in enumerate(tokens):
        if t == "{":
            cnt = 0
            if_split = False
            for j in range(i, len(tokens)):
                if tokens[j] == ";":
                    if_split = True
                elif tokens[j] == "{":
                    cnt += 1
                elif tokens[j] == "}":
                    cnt -= 1
                    if cnt <= 0:
                        if if_split:
                            split_pos.append(i + 1)
                            split_pos.append(j + 1)
                        break
        if t in [";", ":"]:
            if skip_cnt:
                skip_cnt -= 1
                continue
            split_pos.append(i + 1)
        elif t in ["for", "while", "if", "switch"]:
            cnt = 0
            for j in range(i, len(tokens)):
                if tokens[j] == "(":
                    cnt += 1
                elif tokens[j] == ")":
                    cnt -= 1
                    if cnt <= 0:
                        split_pos.append(j + 1)
                        break
            if t == "for":
                skip_cnt = 2
    split_pos = sorted(set(split_pos))
    ret = [0]
    curr_pos = 0
    for i in split_pos:
        if curr_pos + 1 != i:
            ret.append(curr_pos)
        curr_pos = i
    ret.append(curr_pos)
    stmt = []
    for b, e in zip(ret[:-1], ret[1:]):
        if keep_raw:
            stmt.append(_tokens[b: e])
        else:
            stmt.append(tokens[b: e])
    return stmt

if __name__ == "__main__":
    
    import gzip, pickle
    import matplotlib.pyplot as plt
    import tqdm
    
    with gzip.open("../data/oj.pkl.gz", "rb") as f:
        d = pickle.load(f)
        
    code_lens = {}
    stmt_lens = {}
    for src in tqdm.tqdm(d['raw_tr']):
        stmts = brute_stmt_split(src)
        if len(stmts) not in code_lens.keys():
            code_lens[len(stmts)] = 0
        code_lens[len(stmts)] += 1
        for s in stmts:
            if len(s) not in stmt_lens.keys():
                stmt_lens[len(s)] = 0
            stmt_lens[len(s)] += 1
            
    code_lens = sorted(code_lens.items(), key=lambda i: i[0])
    plt.plot([i[0] for i in code_lens], [i[1] for i in code_lens])
    plt.show()
    
    stmt_lens = sorted(stmt_lens.items(), key=lambda i: i[0])
    plt.plot([i[0] for i in stmt_lens], [i[1] for i in stmt_lens])
    plt.show()