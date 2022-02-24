# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:26:29 2020

@author: DrLC
"""

import tqdm

def build_vocab(codes):
    
    vocab_cnt = {"<str>": 0, "<char>": 0, "<int>": 0, "<fp>": 0}
    for c in tqdm.tqdm(codes):
        for t in c:
            if t[0] == '"' and t[-1] == '"':
                vocab_cnt["<str>"] += 1
            elif t[0] == "'" and t[-1] == "'":
                vocab_cnt["<char>"] += 1
            elif t[0] in "0123456789.":
                if 'e' in t.lower() or '.' in t:
                    vocab_cnt["<fp>"] += 1
                else:
                    vocab_cnt["<int>"] += 1
            elif t in vocab_cnt.keys():
                vocab_cnt[t] += 1
            else:
                vocab_cnt[t] = 1
    vocab_cnt = sorted(vocab_cnt.items(), key=lambda x:x[1], reverse=True)
    idx2txt = ["<unk>"] + [it[0] for it in vocab_cnt]
    txt2idx = {}
    for idx in range(len(idx2txt)):
        txt2idx[idx2txt[idx]] = idx
    return idx2txt, txt2idx
    
def text2index(codes, txt2idx):
    
    codes_idx = []
    for c in tqdm.tqdm(codes):
        codes_idx.append([])
        for t in c:
            if t[0] == '"' and t[-1] == '"':
                codes_idx[-1].append(txt2idx["<str>"])
            elif t[0] == "'" and t[-1] == "'":
                codes_idx[-1].append(txt2idx["<char>"])
            elif t[0] in "0123456789.":
                if 'e' in t.lower() or '.' in t:
                    codes_idx[-1].append(txt2idx["<fp>"])
                else:
                    codes_idx[-1].append(txt2idx["<int>"])
            elif t in txt2idx.keys():
                codes_idx[-1].append(txt2idx[t])
            else:
                codes_idx[-1].append(txt2idx["<unk>"])
    return codes_idx