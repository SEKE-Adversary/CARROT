# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:19:07 2020

@author: DrLC
"""

import torch
import random
import copy
import numpy
from copy import deepcopy
import sys
sys.path.append("../")
import pattern
from pycparser import c_parser
from utils import brute_stmt_split

# for checking the validness of adversarial sample
parser = c_parser.CParser()

def raw2x(raws, txt2idx, max_stmt_cnt, max_stmt_len, vocab_size):
    
    xs = []
    ls = []

    for raw in raws:
        stmts = brute_stmt_split(raw) # (raw) to (stmts)
        
        stmts_idx = [] 
        for s in stmts: # (stmts) to (list of ids)
            stmts_idx.append([])
            for t in s:
                if t in txt2idx.keys():
                    stmts_idx[-1].append(txt2idx[t])
                else:
                    stmts_idx[-1].append(txt2idx['<unk>'])
        
        # for each statement
        x = []
        l = []
        for s in stmts_idx[: max_stmt_cnt]:
            if len(s) <= 0:
                continue
            x.append([])
            for t in s[: max_stmt_len]:
                if t >= vocab_size:
                    x[-1].append(txt2idx['<unk>'])
                else:
                    x[-1].append(t)
            if len(s) < max_stmt_len:
                x[-1] += [txt2idx['<pad>']] * (max_stmt_len - len(s))
                l.append(len(s))
            else:
                l.append(max_stmt_len)
        
        # statement count
        if len(x) < max_stmt_cnt:
            l += [-1] * (max_stmt_cnt - len(x))
            x += [[txt2idx['<pad>']] * max_stmt_len] \
                            * (max_stmt_cnt - len(x))
        x = numpy.asarray(x)
        l = numpy.asarray(l)
        xs.append(x)
        ls.append(l)
    
    xs = numpy.asarray(xs)
    ls = numpy.asarray(ls)
    #print(xs.shape) # shape: [1, max_stmt_cnt, max_stmt_len]
    #print(ls.shape) # shape: [1, max_stmt_cnt]

    return xs, ls

def get_batched_data(raws, ys, txt2idx, max_stmt_cnt, max_stmt_len, vocab_size, ids=None):

    xs, ls = raw2x(raws, txt2idx, max_stmt_cnt, max_stmt_len, vocab_size)

    batch = {"x": [], "y": [], "l": [], "raw": [], "id": [], "new_epoch": False}
    batch['x'] = xs
    batch['l'] = ls
    batch['y'] = ys
    batch['id'] = ids
    batch['raw'] = deepcopy(raws)
    
    return batch

def gettensor(batch, device):
    
    return (torch.tensor(batch['x'], dtype=torch.long).to(device),
            torch.tensor(batch['l'], dtype=torch.long).to(device),
            torch.tensor(batch['y'], dtype=torch.long).to(device))

class TokenModifier(object):
    
    def __init__(self, classifier, loss, uids, txt2idx, idx2txt, max_stmt_cnt, max_stmt_len):
        
        self.cl = classifier
        self.loss = loss
        self.txt2idx = txt2idx
        self.idx2txt = idx2txt
        self.max_stmt_cnt = max_stmt_cnt
        self.max_stmt_len = max_stmt_len
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
            if uid in self.txt2idx.keys() and self.txt2idx[uid] not in _uids and uid not in self.forbidden_uid:
                _uids.append(self.txt2idx[uid])
        self._uids = _uids
        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        
    def __gen_uid_mask_on_vocab(self, uids):
        
        _uids = torch.zeros(self.cl.vocab_size)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.vocab_size, 1]).to(self.cl.device)
        return _uids
    
    # return None, None, None when ori_uid is "<unk>" or no uid in "topk" 
    def rename_uid(self, x_raw, y, ori_uid, n_candidate=5):
        
        if ori_uid in self.txt2idx.keys():
            ori_uid_raw = ori_uid
            ori_uid = self.txt2idx[ori_uid]
        else:
            return None, None
        
        batch = get_batched_data([x_raw], [y], self.txt2idx, self.max_stmt_cnt, self.max_stmt_len, self.cl.vocab_size)
        inputs, lens, labels= gettensor(batch, self.cl.device)
        grad = self.cl.grad(inputs, lens, labels, self.loss)
        grad = grad[ori_uid].reshape([1, self.cl.x_size])
        ori_embed = self.cl.embed.weight[ori_uid]\
                    .reshape([1, self.cl.x_size])\
                    .expand([self.cl.vocab_size, self.cl.x_size])
        delta_embed = self.uids * (self.cl.embed.weight - ori_embed)
        delta_embed_len = torch.sqrt(torch.sum(delta_embed*delta_embed, dim=1)) + 1e-5
        inner_prod = torch.sum(grad*delta_embed, dim=1) / delta_embed_len

        _, new_uid_cand =  torch.topk(inner_prod, n_candidate)
        new_uid_cand = new_uid_cand.cpu().numpy()
        new_x_raw, new_x_uid = [], []
        for new_uid in new_uid_cand:
            if not self.uids[new_uid]:
                continue
            if self.idx2txt[new_uid] in x_raw:
                continue
            new_x_uid.append(new_uid)
            new_x_raw.append(copy.deepcopy(x_raw))
            for i in range(len(new_x_raw[-1])):
                if new_x_raw[-1][i] == ori_uid_raw:
                    new_x_raw[-1][i] = self.idx2txt[new_uid]
            try:
                parser.parse(" ".join(new_x_raw[-1]))
            except:
                new_x_uid.pop()
                new_x_raw.pop()

        if len(new_x_uid) == 0:
            return None, None
        while len(new_x_uid) < n_candidate:
            new_x_uid.append(new_x_uid[-1])
            new_x_raw.append(new_x_raw[-1])
        return new_x_raw, new_x_uid

    def rename_uid_random(self, x_raw, ori_uid):
        
        if ori_uid in self.txt2idx.keys():
            ori_uid_raw = ori_uid
            ori_uid = self.txt2idx[ori_uid]
        else:
            return None, None
        
        fail_cnt = 0
        uid_cand = random.sample(self._uids, 1)[0]
        while uid_cand == ori_uid or self.idx2txt[uid_cand] in x_raw:
            fail_cnt += 1 
            uid_cand = random.sample(self._uids, 1)[0]
            if fail_cnt >= 10:  # in case of dead loop
                uid_cand = ori_uid # keep it no changed
                break
        new_x_uid = [uid_cand]
        new_x_raw = [copy.deepcopy(x_raw)]
        for i in range(len(new_x_raw[-1])):
            if new_x_raw[-1][i] == ori_uid_raw:
                new_x_raw[-1][i] = self.idx2txt[uid_cand]
        try:
            parser.parse(" ".join(new_x_raw[-1]))
        except:
            new_x_uid.pop()
            new_x_raw.pop()

        if new_x_uid == []:
            return None, None
        else:
            return new_x_raw, new_x_uid

class InsModifier(object):
    
    def __init__(self, classifier, txt2idx, idx2txt, poses=None):
        
        self.cl = classifier
        self.txt2idx = txt2idx
        self.idx2txt = idx2txt
        if poses is not None: # else you need to call initInsertDict later
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
                   "for ( int i = 0 ; i < 0 ; i ++ ) { }",
                   "while ( false ) ;",
                   "while ( 0 ) ;",
                   "while ( true ) break ;",
                   "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
                   "do { } while ( false ) ;"]
        self.inserts = [insert.split(" ") for insert in inserts]
    
    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0

    # only support one piece of data each time: x is idx-list
    def insert(self, x_raw, n_candidate=5):

        pos_candidates = pattern.InsAddCandidates(self.insertDict) # we handle raw data, do not need to exclude outlier poses
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] # sample max(n, n_candidate) poses

        new_x_raw, new_insertDict = [], []
        for pos in pos_candidates:
            inst = random.sample(self.inserts, 1)[0]
            _insertDict = deepcopy(self.insertDict)
            pattern.InsAdd(_insertDict, pos, inst)
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            try:
                parser.parse(" ".join(_x_raw))
            except:
                continue
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)

        return new_x_raw, new_insertDict

    def remove(self, x_raw, n_candidate=5):

        pos_candidates = pattern.InsDeleteCandidates(self.insertDict) # e.g. [(pos0, 0), (pos0, 1), (pos1, 0), ...]
        n = len(pos_candidates)
        if n_candidate < n:
          candisIdx = random.sample(range(n), n_candidate)
        else:
          candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        new_x_raw, new_insertDict = [], []
        for pos, listIdx in pos_candidates:
            _insertDict = deepcopy(self.insertDict)
            pattern.InsDelete(_insertDict, pos, listIdx)
            _x_raw = pattern.InsResult(x_raw, _insertDict)
            try:
                parser.parse(" ".join(_x_raw))
            except:
                continue
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)

        return new_x_raw, new_insertDict

    def insert_remove_random(self, x_raw):

        new_x_raw, new_insertDict = [], []
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
            try:
                parser.parse(" ".join(_x_raw))
            except:
                fail_cnt += 1
                continue
            new_x_raw.append(_x_raw)
            new_insertDict.append(_insertDict)
            break
        return new_x_raw, new_insertDict

