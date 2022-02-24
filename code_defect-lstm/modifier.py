# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:19:07 2020

@author: DrLC
"""

import torch
import copy
import sys
sys.path.append("/data/fuzy/Uni-MHM/Universal-MHM/preprocess-lstm")
import pattern
import random
from copy import deepcopy

class TokenModifier(object):
    
    def __init__(self, classifier, loss, uids, txt2idx, idx2txt):
        
        self.cl = classifier
        self.loss = loss
        self.txt2idx = txt2idx
        self.idx2txt = idx2txt
        self.__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                             "default", "do", "double", "else", "enum", "extern",
                             "float", "for", "goto", "if", "inline", "int", "long",
                             "register", "restrict", "return", "short", "signed",
                             "sizeof", "static", "struct", "switch", "typedef",
                             "union", "unsigned", "void", "volatile", "while",
                             "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                             "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                             "_Thread_local", "__func__"]
        self.__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
                       ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
                       "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
                       ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
        self.__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
                          "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
                          "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
        self.__special_ids__ = ["main",  # main function
                               "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                               "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                               "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                               "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                               "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                               "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                               "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                               "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                               "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                               "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                               "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                               "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                               "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                               "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                               "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                               "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                               "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                               "mbstowcs", "wcstombs",
                               "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                               "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                               "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                               "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                               "strpbrk" ,"strstr", "strtok", "strxfrm",
                               "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                               "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                               "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                               "iomanip", "iosfwd",
                               "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                               "streamsize", "cout", "cerr", "clog", "cin",
                               "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                               "noshowbase", "showpoint", "noshowpoint", "showpos",
                               "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                               "left", "right", "internal", "dec", "oct", "hex", "fixed",
                               "scientific", "hexfloat", "defaultfloat", "width", "fill",
                               "precision", "endl", "ends", "flush", "ws", "showpoint",
                               "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                               "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                               "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                               "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
        self.forbidden_uid = self.__key_words__ + self.__ops__ + self.__macros__ + self.__special_ids__
        _uids = [txt2idx["<unk>"]]
        for uid in uids:
            if uid in txt2idx.keys() and txt2idx[uid] not in _uids and uid not in self.forbidden_uid:
                _uids.append(txt2idx[uid])
        self._uids = _uids
        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        
    
    def __gen_uid_mask_on_vocab(self, uids):
        
        _uids = torch.zeros(self.cl.vocab_size)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.vocab_size, 1]).cuda()
        return _uids
    
    def __gen_uid_mask_on_seq(self, uids):
        
        _uids = torch.zeros(self.cl.max_len)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.max_len, 1]).cuda()
        return _uids
    
    def rename_uid(self, x, y, x_uid, ori_uid, n_candidate=5):
        
        _x_uid = []
        for i in x_uid:
            if i < self.cl.max_len:
                _x_uid.append(i)
        x_uid = self.__gen_uid_mask_on_seq(_x_uid)
        if ori_uid in self.txt2idx.keys():
            ori_uid = self.txt2idx[ori_uid]
        else:
            ori_uid = self.txt2idx['<unk>']
        _x = torch.tensor(x, dtype=torch.long).cuda().permute([1, 0])
        _y = torch.tensor(y, dtype=torch.long).cuda()
        grad = self.cl.grad(_x, _y, self.loss)[0]
        mean_grad = torch.sum(grad*x_uid, dim=0) / torch.sum(x_uid)
        mean_grad = mean_grad.reshape([1, self.cl.embedding_size])
        ori_embed = self.cl.embedding.weight[ori_uid]\
                    .reshape([1, self.cl.embedding_size])\
                    .expand([self.cl.vocab_size, self.cl.embedding_size])
        delta_embed = self.uids * (self.cl.embedding.weight - ori_embed)
        delta_embed_len = torch.sqrt(torch.sum(delta_embed*delta_embed, dim=1)) + 1e-5
        inner_prod = torch.sum(mean_grad*delta_embed, dim=1) / delta_embed_len
        _, new_uid_cand =  torch.topk(inner_prod, n_candidate)
        new_uid_cand = new_uid_cand.cpu().numpy()
        new_x = []
        for new_uid in new_uid_cand:
            new_x.append(copy.deepcopy(x[0]))
            for i in _x_uid:
                new_x[-1][i] = new_uid
        return new_x, new_uid_cand

    def rename_uid_random(self, x, x_uid, ori_uid):
        
        _x_uid = []
        for i in x_uid:
            if i < self.cl.max_len:
                _x_uid.append(i)
        if ori_uid in self.txt2idx.keys():
            ori_uid = self.txt2idx[ori_uid]
        else:
            ori_uid = self.txt2idx['<unk>']

        fail_cnt = 0
        uid_cand = random.sample(self._uids, 1)[0]
        while uid_cand == ori_uid or uid_cand in x[0][:self.cl.max_len]:
            fail_cnt += 1
            uid_cand = random.sample(self._uids, 1)[0]
            if fail_cnt >= 10:  # in case of dead loop
                uid_cand = ori_uid # keep it no changed
                break
        new_x = [copy.deepcopy(x[0])]
        for i in _x_uid:
            new_x[-1][i] = uid_cand
        return new_x, [uid_cand]

class InsModifier(object):
    
    def __init__(self, classifier, txt2idx, poses=None):
        
        self.cl = classifier
        self.txt2idx = txt2idx
        if poses != None: # else you need to call initInsertDict later
          self.initInsertDict(poses)
        inserts = [
          ";",
          "{ }",
          "printf ( \"\" ) ;",
          "if ( false ) ;",
          "if ( true ) { }",
          "if ( false ) ; else { }",
          "if ( 0 ) ;",
          "if ( false ) { int cnt = 0 ; for ( int i = 0 ; i < 123 ; i ++ ) cnt += 1 ; }"
          "for ( int i = 0 ; i < 100 ; i ++ ) break ;",
          "for ( int i = 0 ; i < 0 ; i ++ ) { }"
          "while ( false ) ;",
          "while ( 0 ) ;",
          "while ( true ) break ;",
          "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
          "do { } while ( false ) ;"]

        self.inserts = [insert.split(" ") for insert in inserts]
    
    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0

    def _insert2idxs(self, insert):
        idxs = []
        for t in insert:
            if self.txt2idx.get(t) is not None:
              idxs.append(self.txt2idx[t])
            else:
              idxs.append(self.txt2idx['<unk>'])
        return idxs

    # only support one piece of data each time: x is idx-list
    def insert(self, x, n_candidate=5):

        pos_candidates = pattern.InsAddCandidates(self.insertDict, self.cl.max_len) # exclude outlier poses
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] # sample max(n, n_candidate) poses

        new_x, new_insertDict = [], [] 
        for pos in pos_candidates:
            inst = random.sample(self.inserts, 1)[0]
            inst_idxs = self._insert2idxs(inst)
            _insertDict = deepcopy(self.insertDict)
            pattern.InsAdd(_insertDict, pos, inst_idxs)
            #print("pos:", pos, "=>", inst, "count", _insertDict["count"])
            _x = pattern.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)

        return new_x, new_insertDict

    # only support one piece of data each time: x is idx-list
    def remove(self, x, n_candidate=5):

        pos_candidates = pattern.InsDeleteCandidates(self.insertDict) # e.g. [(pos0, 0), (pos0, 1), (pos1, 0), ...]
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        new_x, new_insertDict = [], [] 
        for pos, inPosIdx in pos_candidates:
            _insertDict = deepcopy(self.insertDict)
            pattern.InsDelete(_insertDict, pos, inPosIdx)
            #print("pos:", pos, "=>", self.insertDict[pos][inPosIdx], _insertDict["count"])
            _x = pattern.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)

        return new_x, new_insertDict

    def insert_remove_random(self, x):

        new_x, new_insertDict = [], []
        fail_cnt = 0
        while True:
            if fail_cnt >= 10:  # in case of dead loop
                break
            if random.random() > 0.5: # insert
                pos_candidates = pattern.InsAddCandidates(self.insertDict, self.cl.max_len) # exclude outlier poses
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                inst = random.sample(self.inserts, 1)[0]
                inst_idxs = self._insert2idxs(inst)
                _insertDict = deepcopy(self.insertDict)
                pattern.InsAdd(_insertDict, pos_cand, inst_idxs)
            else:
                pos_candidates = pattern.InsDeleteCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                pattern.InsDelete(_insertDict, pos_cand, inPosIdx)
            _x = pattern.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)
            break
        return new_x, new_insertDict
    
def idxs2tokens(idxs, idx2word, unk_idx):
    res = []
    n = len(idx2word)
    for idx in idxs:
      if idx < n:
        res.append(idx2word[idx])
      else:
        res.append(idx2word[unk_idx])
    return res

if __name__ == "__main__":
    
    from dataset import OJ104
    from lstm_classifier import LSTMClassifier, LSTMEncoder
    import argparse
    import pickle, gzip, os, sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    parser.add_argument('-attn', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")
    
    vocab_size = 5000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 104
    max_len = 300

    poj = OJ104(path="../data/oj.pkl.gz", max_len=max_len, vocab_size=vocab_size)
    training_set, valid_set, test_set = poj.train, poj.dev, poj.test
    with gzip.open('../data/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len, attn=opt.attn).cuda()
    classifier.load_state_dict(torch.load('../model/lstm/model-14.pt'))
    
    '''
    m = TokenModifier(classifier, torch.nn.CrossEntropyLoss(), symtab['all'],
                      poj.get_txt2idx(), poj.get_idx2txt())
    
    b = test_set.next_batch(1)
    uid = symtab['te'][b['id'][0]]
    
    x = torch.tensor(b['x'], dtype=torch.long).cuda().permute([1, 0])
    y = torch.tensor(b['y'], dtype=torch.long).cuda()
    print (b['y'][0])
    prob = classifier.prob(x)[0]
    print (int(torch.argmax(prob)), float(prob[b['y'][0]]))
    
    for k in uid.keys():
        if k in m.forbidden_uid:
            continue
        print (k)
        new_x, new_uid_cand = m.rename_uid(b['x'], b['y'], uid[k], k, 5)
        new_prob = classifier.prob(torch.tensor(new_x, dtype=torch.long).cuda().permute([1, 0]))
        for _p, _uid in zip(new_prob, new_uid_cand):
            print (poj.idx2vocab(_uid), _uid, float(_p[b['y'][0]]), int(torch.argmax(_p).cpu()))
        print ()
    '''

    b = test_set.next_batch(1)
    stmt_ins_poses = instab['stmt_te'][b['id'][0]]
    m = InsModifier(classifier, poj.get_txt2idx(), stmt_ins_poses)

    pattern._InsVis(b['raw'][0], stmt_ins_poses)

    x = torch.tensor(b['x'], dtype=torch.long).cuda().permute([1, 0])
    y = torch.tensor(b['y'], dtype=torch.long).cuda()
    print (b['y'][0])
    prob = classifier.prob(x)[0]
    print (int(torch.argmax(prob)), float(prob[b['y'][0]]))
    
    old_x = b['x'][0]
    for _ in range(2):
        new_x, new_insertDict = m.insert(old_x, n_candidate=3)
        feed_new_x = [_x[:classifier.max_len] for _x in new_x]  # this step is very important
        new_prob = classifier.prob(torch.tensor(feed_new_x, dtype=torch.long).cuda().permute([1, 0]))
        for _p, _dict in zip(new_prob, new_insertDict):
            print (float(_p[b['y'][0]]), int(torch.argmax(_p).cpu()))
        #while new_x[-1]=="<unk>": new_x = new_x[:-1]
        #pattern._InsVis(idxs2tokens(new_x[0], poj.get_idx2txt(), poj.get_txt2idx()["<unk>"]), [])
        m.insertDict = new_insertDict[0]
        print ('------------ INSERT -------------', m.insertDict["count"])
    for _ in range(2):
        new_x, new_insertDict = m.remove(old_x, n_candidate=3)
        feed_new_x = [_x[:classifier.max_len] for _x in new_x]  # this step is very important
        new_prob = classifier.prob(torch.tensor(feed_new_x, dtype=torch.long).cuda().permute([1, 0]))
        for _p, _dict in zip(new_prob, new_insertDict):
            print (float(_p[b['y'][0]]), int(torch.argmax(_p).cpu()))
        #while new_x[-1]=="<unk>": new_x = new_x[:-1]
        #pattern._InsVis(idxs2tokens(new_x[0], poj.get_idx2txt(), poj.get_txt2idx()["<unk>"]), [])
        m.insertDict = new_insertDict[0]
        print ('------------ REMOVE -------------', m.insertDict["count"])
