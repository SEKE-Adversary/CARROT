# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:26:02 2020

@author: DrLC
"""

from dataset import CodeChef
from model import BatchProgramClassifier
from modifier import TokenModifier, InsModifier
from utils import get_data
from pycparser import c_parser

import torch
import argparse
import pickle, gzip
import os, sys
import time
from sklearn import metrics

class Attacker(object):
    
    def __init__(self, dataset, symtab, classifier, w2v, parser=None):
        
        if parser is None:
            parser = c_parser.CParser()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    w2v=w2v,
                                    parser=parser)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.w2v = w2v
        self.parser = parser
    
    def attack(self, x_raw, y, uids, n_candidate=100, n_iter=20):
        
        iter = 0
        n_stop = 0
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        seq = ""
        for t in x_raw:
            seq += t + " "
        x_ast = self.parser.parse(seq)
        x_in, y_in = get_data([x_ast], [y], self.w2v)
        old_prob = self.cl.prob(x_in)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                iter += 1
                try:
                    new_x_raw, new_x_ast, new_x_uid = self.tokenM.rename_uid(x_raw, x_ast, y, k, n_candidate)
                except:
                    new_x_raw, new_x_ast, new_x_uid = None, None, None
                if new_x_raw is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                self.cl.batch_size = n_candidate
                self.cl.hidden = self.cl.init_hidden()
                x_in, y_in = get_data(new_x_ast, [y for _ in new_x_ast], self.w2v)
                new_prob = self.cl.prob(x_in)
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x_raw in zip(new_x_uid, new_pred, new_prob, new_x_raw):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.d.idx2vocab(uid), y, old_prob, y, pr[y], p, pr[p]))
                        return True, _x_raw, p.cpu().numpy()
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    x_ast = new_x_ast[new_prob_idx]
                    uids[self.w2v.index2word[int(new_x_uid[new_prob_idx])]] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.d.idx2vocab(int(new_x_uid[new_prob_idx])),
                           y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, x_raw, y
    
    def attack_all(self, n_candidate=100, n_iter=20):
        
        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['raw'][0], b['y'][0], self.syms['te'][b['id'][0]], n_candidate, n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(macroP, macroR, macroF1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class InsAttacker(object):
    
    def __init__(self, dataset, instab, classifier, w2v, parser=None):
        
        if parser is None:
            parser = c_parser.CParser()
        self.insM = InsModifier(classifier=classifier,
                                w2v=w2v,
                                poses=None, # wait to init when attack
                                parser=parser)
        self.cl = classifier
        self.d = dataset
        self.inss = instab
        self.w2v = w2v
        self.parser = parser
    
    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_candidate=100, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        seq = " ".join(x_raw)
        x_ast = self.parser.parse(seq)
        x_in, y_in = get_data([x_ast], [y], self.w2v)
        old_prob = self.cl.prob(x_in)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_x_ast_del, new_insertDict_del = self.insM.remove(x_raw, n_candidate_del)
            new_x_raw_add, new_x_ast_add, new_insertDict_add = self.insM.insert(x_raw, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_x_ast = new_x_ast_del + new_x_ast_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x_raw == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            self.cl.batch_size = len(new_x_raw)     # len(new_x) could be smaller than n_candidate
            self.cl.hidden = self.cl.init_hidden()
            x_in, y_in = get_data(new_x_ast, [y for _ in new_x_ast], self.w2v)
            new_prob = self.cl.prob(x_in)
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x_raw in zip(new_insertDict, new_pred, new_prob, new_x_raw):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y, old_prob, y, pr[y], p, pr[p]))
                    return True, _x_raw, p.cpu().numpy()

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                        y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= len(new_x_raw):    # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print ("FAIL!")
        return False, x_raw, y
    
    def attack_all(self, n_candidate=100, n_iter=20):

        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_te'][b['id'][0]], n_candidate, n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(macroP, macroR, macroF1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class AttackerRandom(object):
    
    def __init__(self, dataset, symtab, classifier, w2v, parser=None):
        
        if parser is None:
            parser = c_parser.CParser()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    w2v=w2v,
                                    parser=parser)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
        self.w2v = w2v
        self.parser = parser
    
    def attack(self, x_raw, y, uids, n_iter=20):
        
        iter = 0
        n_stop = 0
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        seq = ""
        for t in x_raw:
            seq += t + " "
        x_ast = self.parser.parse(seq)
        x_in, y_in = get_data([x_ast], [y], self.w2v)
        old_prob = self.cl.prob(x_in)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                iter += 1
                new_x_raw, new_x_ast, new_x_uid = self.tokenM.rename_uid_random(x_raw, x_ast, k)
                if new_x_raw is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                self.cl.batch_size = len(new_x_raw)
                self.cl.hidden = self.cl.init_hidden()
                x_in, y_in = get_data(new_x_ast, [y for _ in new_x_ast], self.w2v)
                new_prob = self.cl.prob(x_in)
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x_raw in zip(new_x_uid, new_pred, new_prob, new_x_raw):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.d.idx2vocab(uid), y, old_prob, y, pr[y], p, pr[p]))
                        return True, _x_raw, p.cpu().numpy()
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    x_ast = new_x_ast[new_prob_idx]
                    uids[self.w2v.index2word[int(new_x_uid[new_prob_idx])]] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.d.idx2vocab(int(new_x_uid[new_prob_idx])),
                           y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, x_raw, y
    
    def attack_all(self, n_iter=20):
        
        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['raw'][0], b['y'][0], self.syms['te'][b['id'][0]], n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(macroP, macroR, macroF1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class InsAttackerRandom(object):
    
    def __init__(self, dataset, instab, classifier, w2v, parser=None):
        
        if parser is None:
            parser = c_parser.CParser()
        self.insM = InsModifier(classifier=classifier,
                                w2v=w2v,
                                poses=None, # wait to init when attack
                                parser=parser)
        self.cl = classifier
        self.d = dataset
        self.inss = instab
        self.w2v = w2v
        self.parser = parser
    
    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        seq = " ".join(x_raw)
        x_ast = self.parser.parse(seq)
        x_in, y_in = get_data([x_ast], [y], self.w2v)
        old_prob = self.cl.prob(x_in)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw, torch.argmax(old_prob).cpu().numpy()
        old_prob = old_prob[y]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            new_x_raw, new_x_ast, new_insertDict = self.insM.insert_remove_random(x_raw)
            if new_x_raw == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            self.cl.batch_size = len(new_x_raw)
            self.cl.hidden = self.cl.init_hidden()
            x_in, y_in = get_data(new_x_ast, [y for _ in new_x_ast], self.w2v)
            new_prob = self.cl.prob(x_in)
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x_raw in zip(new_insertDict, new_pred, new_prob, new_x_raw):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y, old_prob, y, pr[y], p, pr[p]))
                    return True, _x_raw, p.cpu().numpy()

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                        y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= 10:
                iter = n_iter
                break
        print ("FAIL!")
        return False, x_raw, y
    
    def attack_all(self, n_iter=20):

        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_te'][b['id'][0]], n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            precision = metrics.precision_score(trues, preds, average='macro')
            recall = metrics.recall_score(trues, preds, average='macro')
            f1 = metrics.f1_score(trues, preds, average='macro')
            print("\t(macroP, macroR, macroF1) = (%.3f, %.3f, %.3f)" % (precision, recall, f1))
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))
   
if __name__ == "__main__":

    import pycparser
    from gensim.models.word2vec import Word2Vec
    import numpy as np
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-gpu', required=True)
    opt = arg_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    root = '../data_defect/'
    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    
    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 4
    EPOCHS = 15
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    
    cc = CodeChef(path="../data_defect/codechef.pkl.gz")
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test
    with gzip.open('../data_defect/codechef_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_defect/codechef_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    
    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS+1,
                                   ENCODE_DIM, LABELS, 1, USE_GPU, embeddings).cuda()
    model.load_state_dict(torch.load('../model_defect/astnn/model.pt'))
    
    atk = Attacker(cc, symtab, model, word2vec)
    atk.attack_all(40, 50)

    #atk = InsAttacker(cc, instab, model, word2vec)
    #atk.attack_all(40, 20)

    #atk = AttackerRandom(cc, symtab, model, word2vec)
    #atk.attack_all(100)

    #atk = InsAttackerRandom(cc, instab, model, word2vec)
    #atk.attack_all(20)