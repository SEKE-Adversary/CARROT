# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:19:07 2020

@author: DrLC
"""

from dataset import OJ104
from model import BatchProgramCC
from utils import get_data
from torch.autograd import Variable
from gensim.models.word2vec import Word2Vec

import torch
import argparse
import pickle, gzip
import os, sys
sys.path.append("../preprocess-lstm")
import pattern
import time, random
import copy, pycparser
import numpy as np
from copy import deepcopy

class TokenModifier(object):
    
    def __init__(self, classifier, loss, uids, w2v, parser=None):
        
        self.cl = classifier
        self.loss = loss
        self.w2v = w2v
        if parser is None:
            parser = pycparser.c_parser.CParser()
        self.parser = parser
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
        _uids = []
        for uid in uids:
            if uid in self.w2v.vocab.keys() and self.w2v.vocab[uid].index not in _uids and uid not in self.forbidden_uid:
                _uids.append(self.w2v.vocab[uid].index)
        self._uids = _uids
        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        
    def __gen_uid_mask_on_vocab(self, uids):
        
        _uids = torch.zeros(self.cl.vocab_size)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.vocab_size, 1]).cuda()
        return _uids
    
    def rename_uid(self, x_raw, x_ast, x2_ast, y, ori_uid, n_candidate=5):
        
        if ori_uid in self.w2v.vocab.keys():
            ori_uid_raw = ori_uid
            ori_uid = self.w2v.vocab[ori_uid].index
        else:
            return None, None, None
        _x, _y = get_data([x_ast], [y], self.w2v)
        _x2, _y = get_data([x2_ast], [y], self.w2v)
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        grad = self.cl.grad(_x, _x2, Variable(torch.LongTensor(_y).cuda()), self.loss)
        grad = grad[ori_uid].reshape([1, self.cl.embedding_dim])
        ori_embed = self.cl.encoder.embedding.weight[ori_uid]\
                    .reshape([1, self.cl.embedding_dim])\
                    .expand([self.cl.vocab_size, self.cl.embedding_dim])
        delta_embed = self.uids * (self.cl.encoder.embedding.weight - ori_embed)
        delta_embed_len = torch.sqrt(torch.sum(delta_embed*delta_embed, dim=1)) + 1e-5
        inner_prod = torch.sum(grad*delta_embed, dim=1) / delta_embed_len
        _, new_uid_cand =  torch.topk(inner_prod, n_candidate)
        new_uid_cand = new_uid_cand.cpu().numpy()
        new_x_raw, new_x_ast, new_x_uid = [], [], []
        for new_uid in new_uid_cand:
            if not self.uids[new_uid]:
                continue
            new_x_uid.append(new_uid)
            new_x_raw.append(copy.deepcopy(x_raw))
            for i in range(len(new_x_raw[-1])):
                if new_x_raw[-1][i] == ori_uid_raw:
                    new_x_raw[-1][i] = self.w2v.index2word[new_uid]
            seq = ""
            for t in new_x_raw[-1]:
                seq += t + " "
            new_x_ast.append(self.parser.parse(seq))
        if len(new_x_uid) == 0:
            return None, None, None
        while len(new_x_ast) < n_candidate:
            new_x_uid.append(new_x_uid[-1])
            new_x_raw.append(new_x_raw[-1])
            new_x_ast.append(new_x_ast[-1])
        return new_x_raw, new_x_ast, new_x_uid

    def rename_uid_random(self, x_raw, ori_uid):
        
        if ori_uid in self.w2v.vocab.keys():
            ori_uid_raw = ori_uid
            ori_uid = self.w2v.vocab[ori_uid].index
        else:
            return None, None, None

        fail_cnt = 0
        uid_cand = random.sample(self._uids, 1)[0]
        while uid_cand == ori_uid or self.w2v.index2word[uid_cand] in x_raw:
            fail_cnt += 1 
            uid_cand = random.sample(self._uids, 1)[0]
            if fail_cnt >= 10:  # in case of dead loop
                uid_cand = ori_uid # keep it no changed
                break
        new_x_uid = [uid_cand]
        new_x_raw = [copy.deepcopy(x_raw)]
        for i in range(len(new_x_raw[-1])):
            if new_x_raw[-1][i] == ori_uid_raw:
                new_x_raw[-1][i] = self.w2v.index2word[uid_cand]
        new_x_ast = [self.parser.parse(" ".join(new_x_raw[-1]))]
        return new_x_raw, new_x_ast, new_x_uid

class InsModifier(object):
    
    def __init__(self, classifier, w2v, poses=None, parser=None):
        
        self.cl = classifier
        self.txt2idx = dict([(k, w2v.vocab[k].index) for k in w2v.vocab.keys()])
        if poses != None: # else you need to call initInsertDict later
          self.initInsertDict(poses)
        inserts = [";",
                   "{ }",
                   "printf ( \"\" ) ;",
                   "if ( false ) ;",
                   "if ( true ) { }",
                   "if ( false ) ; else { }",
                   "if ( 0 ) ;",
                   "if ( false ) { int cnt = 0 ; for ( int i = 0 ; i < 123 ; i ++ ) cnt += 1 ; }",
                   "for ( int i = 0 ; i < 100 ; i ++ ) break ;",
                   "for ( int i = 0 ; i < 0 ; i ++ ) { }"
                   "while ( false ) ;",
                   "while ( 0 ) ;",
                   "while ( true ) break ;",
                   "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
                   "do { } while ( false ) ;"]
        self.inserts = [insert.split(" ") for insert in inserts]
        if parser is None:
            self.parser = pycparser.c_parser.CParser()
        else:
            self.parser = parser
    
    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0

    def get_ast(self, x_raw):
        raw = ' '.join(x_raw)
        try:
          ast = self.parser.parse(raw)
          return ast
        except:
          return None

    # only support one piece of data each time: x is idx-list
    def insert(self, x_raw, x2_raw, n_candidate=5):

        pos_candidates = pattern.InsAddCandidates(self.insertDict) # exclude outlier poses
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] # sample max(n, n_candidate) poses

        new_x_raw, new_x_ast, new_x2_ast, new_insertDict = [], [], [], []
        x2_ast = self.get_ast(x2_raw)
        for pos in pos_candidates:
            inst = random.sample(self.inserts, 1)[0]
            _insertDict = deepcopy(self.insertDict)
            pattern.InsAdd(_insertDict, pos, inst)
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            _x_ast = self.get_ast(_x_raw)
            if _x_ast and x2_ast:  # omit un-parsed ast in case
              #print("pos:", pos, "=>", inst, "count", _insertDict["count"])
              new_x_raw.append(_x_raw)
              new_x_ast.append(_x_ast)
              new_x2_ast.append(deepcopy(x2_ast))
              new_insertDict.append(_insertDict)

        return new_x_raw, new_x_ast, new_x2_ast, new_insertDict

    # only support one piece of data each time: x is idx-list
    def remove(self, x_raw, x2_raw, n_candidate=5):

        pos_candidates = pattern.InsDeleteCandidates(self.insertDict) # exclude outlier poses
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] # sample max(n, n_candidate) poses

        new_x_raw, new_x_ast, new_x2_ast, new_insertDict = [], [], [], []
        x2_ast = self.get_ast(x2_raw)
        for pos, inPosIdx in pos_candidates:
            _insertDict = deepcopy(self.insertDict)
            pattern.InsDelete(_insertDict, pos, inPosIdx)
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            _x_ast = self.get_ast(_x_raw)
            if _x_ast and x2_ast:  # omit un-parsed ast in case
              #print("pos:", pos, "=>", inst, "count", _insertDict["count"])
              new_x_raw.append(_x_raw)
              new_x_ast.append(_x_ast)
              new_x2_ast.append(deepcopy(x2_ast))
              new_insertDict.append(_insertDict)

        return new_x_raw, new_x_ast, new_x2_ast, new_insertDict

    def insert_remove_random(self, x_raw, x2_raw):

        new_x_raw, new_x_ast, new_x2_ast, new_insertDict = [], [], [], []
        fail_cnt = 0
        while True:
            if fail_cnt >= 10:  # in case of dead loop
                break
            if random.random() > 0.5: # insert
                pos_candidates = pattern.InsAddCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                inst = random.sample(self.inserts, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                pattern.InsAdd(_insertDict, pos_cand, inst)
            else:
                pos_candidates = pattern.InsDeleteCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                pattern.InsDelete(_insertDict, pos_cand, inPosIdx)
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            _x_ast = self.get_ast(_x_raw)
            _x2_ast = self.get_ast(x2_raw)
            if _x_ast and _x2_ast:  # omit un-parsed ast in case
                new_x_raw.append(_x_raw)
                new_x_ast.append(_x_ast)
                new_x2_ast.append(_x2_ast)
                new_insertDict.append(_insertDict)
                break
            else:
                fail_cnt += 1
        return new_x_raw, new_x_ast, new_x2_ast, new_insertDict

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-gpu', required=True)
    opt = arg_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    root = '../data_clone/'
    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    EPOCHS = 15
    BATCH_SIZE = 32
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    
    poj = OJ104(path="../data_clone/oj.pkl.gz")
    training_set, valid_set, test_set = poj.train, poj.dev, poj.test
    with gzip.open('../data_clone/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_clone/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS+1,
                                   ENCODE_DIM, LABELS, 1, USE_GPU, embeddings).cuda()
    model.load_state_dict(torch.load('../model_clone/astnn/model.pt'))
    parser = pycparser.c_parser.CParser()

    b = test_set.next_batch(1)
    stmt_ins_poses = instab['stmt_te'][b['id1'][0]]
    m = InsModifier(model, word2vec, stmt_ins_poses, parser=parser)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    pattern._InsVis(b['raw1'][0], stmt_ins_poses)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    pattern._InsVis(b['raw2'][0], [])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    x_raw, x2_raw, y = b['raw1'][0], b['raw2'][0], b['y'][0]
    x_ast = parser.parse(" ".join(x_raw))
    x2_ast = parser.parse(" ".join(x2_raw))
    x_in, _ = get_data([x_ast], [y], word2vec)
    x2_in, _ = get_data([x2_ast], [y], word2vec)
    print (b['y'][0])
    model.batch_size = 1
    model.hidden = model.init_hidden()
    prob = model.prob(x_in, x2_in)[0]
    print (int(torch.argmax(prob)), float(prob[b['y'][0]]))
    print()
    
    for _ in range(3):
        new_x_raw, new_x_ast, new_x2_ast, new_insertDict = m.insert(x_raw, x2_raw, n_candidate=3)
        for _x_ast, _x2_ast, _insertDict in zip(new_x_ast, new_x2_ast, new_insertDict):
            x_in, _ = get_data([_x_ast], [y], word2vec)
            x2_in, _ = get_data([_x2_ast], [y], word2vec)
            model.batch_size = 1
            model.hidden = model.init_hidden()
            prob = model.prob(x_in, x2_in)[0]
            print (int(torch.argmax(prob)), float(prob[b['y'][0]]))
        if new_x_raw!=[]:
            m.insertDict = new_insertDict[0]
        print ('------------ INSERT -------------', m.insertDict["count"])
    for _ in range(2):
        new_x_raw, new_x_ast, new_x2_ast, new_insertDict = m.remove(x_raw, x2_raw, n_candidate=3)
        for _x_ast, _x2_ast, _insertDict in zip(new_x_ast, new_x2_ast, new_insertDict):
            x_in, _ = get_data([_x_ast], [y], word2vec)
            x2_in, _ = get_data([_x2_ast], [y], word2vec)
            model.batch_size = 1
            model.hidden = model.init_hidden()
            prob = model.prob(x_in, x2_in)[0]
            print (int(torch.argmax(prob)), float(prob[b['y'][0]]))
        if new_x_raw!=[]:
            m.insertDict = new_insertDict[0]
        print ('------------ REMOVE -------------', m.insertDict["count"])