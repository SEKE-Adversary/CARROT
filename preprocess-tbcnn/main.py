# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:26:37 2021

@author: DrLC
"""

from tree import extract_ast, SingleNode

import pycparser
import pickle, gzip
import tqdm

if __name__ == "__main__":
    
    src_path = "../data/oj.pkl.gz"
    tgt_path = "../data/oj_ast_graph.pkl.gz"
    
    with gzip.open(src_path, "rb") as f:
        d = pickle.load(f)
    raw_tr = d['raw_tr']
    raw_te = d['raw_te']
    y_tr = d['y_tr']
    y_te = d['y_te']
    assert len(raw_tr) == len(y_tr)
    assert len(raw_te) == len(y_te)
    
    parser = pycparser.c_parser.CParser()
    
    node_tr, begin_tr, end_tr = [], [], []
    for raw in tqdm.tqdm(raw_tr):
        ast = parser.parse(" ".join(raw))
        n, b, e = extract_ast(ast, 0, [SingleNode(ast).get_token()], [], [])
        assert len(b) == len(e)
        assert len(n) == len(b) + 1
        node_tr.append(n)
        begin_tr.append(b)
        end_tr.append(e)

            
    node_te, begin_te, end_te = [], [], []
    for raw in tqdm.tqdm(raw_te):
        ast = parser.parse(" ".join(raw))
        n, b, e = extract_ast(ast, 0, [SingleNode(ast).get_token()], [], [])
        assert len(b) == len(e)
        assert len(n) == len(b) + 1
        node_te.append(n)
        begin_te.append(b)
        end_te.append(e)
        
    node_cnt = {}
    for nodes in node_tr:
        for node in nodes:
            if node in node_cnt.keys():
                node_cnt[node] += 1
            else:
                node_cnt[node] = 1
    node_cnt = sorted(node_cnt.items(), key=lambda item: item[1], reverse=True)
    idx2node = ["<unk>"] + [it[0] for it in node_cnt]
    node2idx = {}
    for idx in range(len(idx2node)):
        node2idx[idx2node[idx]] = idx
        assert node2idx[idx2node[idx]] == idx
    
    for i in tqdm.tqdm(range(len(node_tr))):
        for j in range(len(node_tr[i])):
            node_tr[i][j] = node2idx[node_tr[i][j]]
    for i in tqdm.tqdm(range(len(node_te))):
        for j in range(len(node_te[i])):
            if node_te[i][j] in node2idx.keys():
                node_te[i][j] = node2idx[node_te[i][j]]
            else:
                node_te[i][j] = node2idx["<unk>"]
            
    dataset = {"raw_tr": raw_tr, "y_tr": y_tr, "node_tr": node_tr,
               "begin_tr": begin_tr, "end_tr": end_tr,
               "raw_te": raw_te, "y_te": y_te, "node_te": node_te,
               "begin_te": begin_te, "end_te": end_te,
               "node2idx": node2idx, "idx2node": idx2node}
    with gzip.open(tgt_path, "wb") as f:
        pickle.dump(dataset, f)
