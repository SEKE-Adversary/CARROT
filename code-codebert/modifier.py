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

# for checking the validness of adversarial sample
parser = c_parser.CParser()

def raw2x(raws, txt2idx):
    "here we dont convert raw to ids in fact, but replace OoVs as '<unk>'"

    xs = []

    for raw in raws:
        xs.append([])
        for token in raw:
            if token in txt2idx.keys():
                xs[-1].append(token)
            else:
                xs[-1].append("<unk>")

    return xs

def get_batched_data(raws, ys, txt2idx, ids=None):

    xs = raw2x(raws, txt2idx)

    batch = {"x": [], "y": [], "raw": [], "id": [], "new_epoch": False}
    batch['x'] = xs # is still token list, but with certain '<unk>'s
    batch['x'] = [" ".join(x) for x in xs]
    batch['y'] = ys
    batch['id'] = ids
    batch['raw'] = deepcopy(raws)
    
    return batch

class TokenModifier(object):
    
    def __init__(self, classifier, loss, uids, txt2idx, idx2txt):
        
        self.cl = classifier
        self.loss = loss

        # poj's vocab, not codebert's vocab
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
        
        _uids = []
        # check every subtoken whether or not it can be treated as an valid uid
        for subtoken_idx in range(self.cl.vocab_size):
            subtoken = self.cl.tokenizer.convert_ids_to_tokens(subtoken_idx)
            assert isinstance(subtoken, str)
            if subtoken in [self.cl.tokenizer.bos_token, self.cl.tokenizer.eos_token,
                                self.cl.tokenizer.sep_token, self.cl.tokenizer.pad_token,
                                self.cl.tokenizer.unk_token, self.cl.tokenizer.cls_token,
                                self.cl.tokenizer.mask_token]:
                continue
            if not subtoken.startswith('Ġ'):
                continue
            # Ġxxxx subtoken is the start token of the new word, we only take these subtokens as candidates
            clear_subtoken = subtoken[1:]
            if clear_subtoken=="":
                continue
            if clear_subtoken[0] in '0987654321':
                continue

            for uid in uids:
                if uid in self.txt2idx.keys() and \
                   clear_subtoken in uid and \
                   uid not in self.forbidden_uid and \
                   subtoken_idx not in _uids and \
                   clear_subtoken not in self.forbidden_uid:
                    _uids.append(subtoken_idx)
                    break
        
        self._uids = _uids
        #print([self.cl.tokenizer.convert_ids_to_tokens(i) for i in self._uids])
        #input()

        self.uids = self.__gen_uid_mask_on_vocab(_uids)
        
    def __gen_uid_mask_on_vocab(self, uids):
    
        _uids = torch.zeros(self.cl.vocab_size)
        _uids.index_put_([torch.LongTensor(uids)], torch.Tensor([1 for _ in uids]))
        _uids = _uids.reshape([self.cl.vocab_size, 1]).to(self.cl.device)
        return _uids
    
    # return None, None, None when ori_uid is "<unk>" or no uid in "topk" 
    def rename_uid(self, x_raw, y, ori_uid_raw, n_candidate=5):
       
        # uid is token in dataset.vocab, not token in codebert.vocab
        Gori_uid_raw = 'Ġ' + ori_uid_raw

        Gori_uid_idx = self.cl.tokenizer.convert_tokens_to_ids(Gori_uid_raw)
        if not self.uids[Gori_uid_idx]:
            return None, None

        batch = get_batched_data([x_raw], [y], self.txt2idx)
        grad = self.cl.grad(batch['x'], batch['y'])
        grad = grad[Gori_uid_idx].reshape([1, self.cl.x_size])
        ori_embed = self.cl.embed.weight[Gori_uid_idx]\
                    .reshape([1, self.cl.x_size])\
                    .expand([self.cl.vocab_size, self.cl.x_size])
        delta_embed = self.uids * (self.cl.embed.weight - ori_embed)
        delta_embed_len = torch.sqrt(torch.sum(delta_embed*delta_embed, dim=1)) + 1e-5
        inner_prod = torch.sum(grad*delta_embed, dim=1) / delta_embed_len

        _, Gnew_uid_cand =  torch.topk(inner_prod, n_candidate)
        Gnew_uid_cand = Gnew_uid_cand.cpu().numpy()
        new_x_raw, new_x_uid = [], []
        for Gnew_uid_idx in Gnew_uid_cand:
            if not self.uids[Gnew_uid_idx]:
                continue
            Gnew_uid_raw = self.cl.tokenizer.convert_ids_to_tokens(int(Gnew_uid_idx))
            new_uid_raw = Gnew_uid_raw[1:]
            if new_uid_raw in x_raw:
                continue
            new_x_uid.append(new_uid_raw)
            new_x_raw.append(copy.deepcopy(x_raw))
            for i in range(len(new_x_raw[-1])):
                if new_x_raw[-1][i] == ori_uid_raw:
                    new_x_raw[-1][i] = new_uid_raw
            try:
                parser.parse(" ".join(new_x_raw[-1]))
            except:
                new_x_uid.pop()
                new_x_raw.pop()

        if len(new_x_uid) == 0:
            #print('!!!! NO valid candidate !!!!')
            return None, None
        while len(new_x_uid) < n_candidate:
            new_x_uid.append(new_x_uid[-1])
            new_x_raw.append(new_x_raw[-1])
        return new_x_raw, new_x_uid

    def rename_uid_random(self, x_raw, ori_uid_raw):
        
        # uid is token in dataset.vocab, not token in codebert.vocab
        Gori_uid_raw = 'Ġ' + ori_uid_raw

        Gori_uid_idx = self.cl.tokenizer.convert_tokens_to_ids(Gori_uid_raw)
        if not self.uids[Gori_uid_idx]:
            return None, None
        
        fail_cnt = 0
        Guid_cand_idx = random.sample(self._uids, 1)[0]
        Guid_cand_raw = self.cl.tokenizer.convert_ids_to_tokens(Guid_cand_idx)
        uid_cand_raw = Guid_cand_raw[1:]
        while Guid_cand_idx == Gori_uid_idx or uid_cand_raw in x_raw:
            fail_cnt += 1 
            Guid_cand_idx = random.sample(self._uids, 1)[0]
            Guid_cand_raw = self.cl.tokenizer.convert_ids_to_tokens(Guid_cand_idx)
            uid_cand_raw = Guid_cand_raw[1:]
            if fail_cnt >= 10:  # in case of dead loop
                Guid_cand_idx = Gori_uid_idx # keep it no changed
                Guid_cand_raw = Gori_uid_raw
                uid_cand_raw = ori_uid_raw
                break
        new_x_uid = [uid_cand_raw]
        new_x_raw = [copy.deepcopy(x_raw)]
        for i in range(len(new_x_raw[-1])):
            if new_x_raw[-1][i] == ori_uid_raw:
                new_x_raw[-1][i] = uid_cand_raw
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

