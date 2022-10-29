# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:26:02 2020

@author: DrLC
"""

from dataset import OJ104
from model import BatchProgramCC
from modifier import TokenModifier, InsModifier
from utils import get_data
from pycparser import c_parser
from sklearn.metrics import precision_recall_fscore_support

import torch
import argparse
import pickle, gzip
import os, sys
import time
import copy

class AdversarialTrainingAttacker(object):
    
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
    
    def attack(self, x_raw, x2_raw, y, uids, n_candidate=100, n_iter=20, relax=1):
        
        ori_x_raw = copy.deepcopy(x_raw)
        iter = 0
        n_stop = 0
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        seq = ""
        for t in x_raw:
            seq += t + " "
        x_ast = self.parser.parse(seq)
        seq = ""
        for t in x2_raw:
            seq += t + " "
        x2_ast = self.parser.parse(seq)
        x_in, y_in = get_data([x_ast], [y], self.w2v)
        x2_in, y_in = get_data([x2_ast], [y], self.w2v)
        old_prob = self.cl.prob(x_in, x2_in)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw, x2_raw, 0
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
                new_x_raw, new_x_ast, new_x_uid = self.tokenM.rename_uid(x_raw, x_ast, x2_ast,
                                                                         y, k, n_candidate)
                if new_x_raw is None:
                    n_stop += 1
                    print ("skip unk\t%s" % k)
                    continue
                self.cl.batch_size = n_candidate
                self.cl.hidden = self.cl.init_hidden()
                x_in, y_in = get_data(new_x_ast, [y for _ in new_x_ast], self.w2v)
                x2_in, y_in = get_data([x2_ast for _ in new_x_ast],
                                       [y for _ in new_x_ast], self.w2v)
                new_prob = self.cl.prob(x_in, x2_in)
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _raw in zip(new_x_uid, new_pred, new_prob, new_x_raw):
                    if p != y:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.w2v.index2word[uid], y, old_prob, y, pr[y], p, pr[p]))
                        return True, _raw, x2_raw, 1
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] * relax < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    x_ast = new_x_ast[new_prob_idx]
                    uids[self.w2v.index2word[int(new_x_uid[new_prob_idx])]] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.w2v.index2word[int(new_x_uid[new_prob_idx])],
                           y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, ori_x_raw, x2_raw, 2
    
    def attack_all(self, n_candidate=100, n_iter=20, relax=1, res_save=None, adv_sample_size=5000):
        
        n_succ, n_total = 0, 0
        total_time = 0
        adv_raw1s, adv_raw2s, adv_labels = [], [], []
        fail_pred_raw1s, fail_pred_raw2s, fail_pred_labels = [], [], []
        st_time = time.time()
        for i in range(self.d.train.get_size()):
            if len(adv_raw1s) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)
            print ("\t%d/%d\tID = (%d, %d)\tY = %d" %
                   (i+1, self.d.train.get_size(), b['id1'][0], b['id2'][0], b['y'][0]))
            start_time = time.time()
            tag, x, x2, typ = self.attack(b['raw1'][0], b['raw2'][0], b['y'][0],
                                               self.syms['tr'][b['id1'][0]], n_candidate, n_iter, relax)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_raw1s.append(x)
                fail_pred_raw2s.append(x2)
                fail_pred_labels.append(int(b['y'][0]))
            if typ == 1:
                adv_raw1s.append(x)
                adv_raw2s.append(x2)
                adv_labels.append(int(b['y'][0]))
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            n_total += 1
            if len(adv_raw1s) % 500 == 0 and res_save is not None:
                partial_res_save = res_save.split('/')
                name_split = partial_res_save[-1].split('.')
                name_split[0] += "_" + str(len(adv_raw1s))
                partial_res_save[-1] = ".".join(name_split)
                partial_res_save = "/".join(partial_res_save)
                with gzip.open(partial_res_save, "wb") as f:
                    pickle.dump({"fail_pred_raw1s": fail_pred_raw1s, 
                                 "fail_pred_raw2s": fail_pred_raw2s, 
                                 "fail_pred_labels": fail_pred_labels,
                                 "adv_raw1s": adv_raw1s,
                                 "adv_raw2s": adv_raw2s,
                                 "adv_labels": adv_labels}, f)
        if res_save is not None:
            print ("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_raw1s), len(fail_pred_raw1s)))
            with gzip.open(res_save, "wb") as f:
                pickle.dump({"fail_pred_raw1s": fail_pred_raw1s, 
                             "fail_pred_raw2s": fail_pred_raw2s, 
                             "fail_pred_labels": fail_pred_labels,
                             "adv_raw1s": adv_raw1s,
                             "adv_raw2s": adv_raw2s,
                             "adv_labels": adv_labels}, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/n_total))

class AdversarialTrainingInsAttacker(object):
    
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
    def attack(self, x_raw, x2_raw, y, poses, n_candidate=100, n_iter=20, relax=1):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        self.cl.batch_size = 1
        self.cl.hidden = self.cl.init_hidden()
        x_ast = self.parser.parse(" ".join(x_raw))
        x2_ast = self.parser.parse(" ".join(x2_raw))
        x_in, _ = get_data([x_ast], [y], self.w2v)
        x2_in, _ = get_data([x2_ast], [y], self.w2v)
        old_prob = self.cl.prob(x_in, x2_in)[0]
        if torch.argmax(old_prob) != y:
            print ("SUCC! Original mistake.")
            return True, x_raw, x2_raw, 0
        old_prob = old_prob[y]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_x_ast_del, new_x2_ast_del, new_insertDict_del = self.insM.remove(x_raw, x2_raw, n_candidate_del)
            new_x_raw_add, new_x_ast_add, new_x2_ast_add, new_insertDict_add = self.insM.insert(x_raw, x2_raw, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_x_ast = new_x_ast_del + new_x_ast_add
            new_x2_ast = new_x2_ast_del + new_x2_ast_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x_raw == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            self.cl.batch_size = len(new_x_raw)     # len(new_x) could be smaller than n_candidate
            self.cl.hidden = self.cl.init_hidden()
            x_in, _ = get_data(new_x_ast, [y for _ in new_x_ast], self.w2v)
            x2_in, _ = get_data(new_x2_ast, [y for _ in new_x2_ast], self.w2v)
            new_prob = self.cl.prob(x_in, x2_in)
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x_raw in zip(new_insertDict, new_pred, new_prob, new_x_raw):
                if p != y:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], y, old_prob, y, pr[y], p, pr[p]))
                    return True, _x_raw, x2_raw, 1

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] * relax < old_prob:
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
        return False, x_raw, x2_raw, 2
    
    def attack_all(self, n_candidate=100, n_iter=20, relax=1, res_save=None, adv_sample_size=5000):

        n_succ, n_total = 0, 0
        total_time = 0
        adv_raw1s, adv_raw2s, adv_labels = [], [], []
        fail_pred_raw1s, fail_pred_raw2s, fail_pred_labels = [], [], []
        st_time = time.time()
        for i in range(self.d.train.get_size()):
            if len(adv_raw1s) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)
            print ("\t%d/%d\tID = (%d, %d)\tY = %d" %
                   (i+1, self.d.train.get_size(), b['id1'][0], b['id2'][0], b['y'][0]))
            start_time = time.time()
            tag, x, x2, typ = self.attack(b['raw1'][0], b['raw2'][0], b['y'][0], 
                                           self.inss['stmt_tr'][b['id1'][0]], n_candidate, n_iter, relax)
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_raw1s.append(x)
                fail_pred_raw2s.append(x2)
                fail_pred_labels.append(int(b['y'][0]))
            if typ == 1:
                adv_raw1s.append(x)
                adv_raw2s.append(x2)
                adv_labels.append(int(b['y'][0]))
            if n_succ <= 0:
                print ("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                       % (n_succ/(i+1)), flush=True)
            else:
                print ("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                       % (n_succ/(i+1), total_time/n_succ), flush=True)
            n_total += 1
            if len(adv_raw1s) % 500 == 0 and res_save is not None:
                partial_res_save = res_save.split('/')
                name_split = partial_res_save[-1].split('.')
                name_split[0] += "_" + str(len(adv_raw1s))
                partial_res_save[-1] = ".".join(name_split)
                partial_res_save = "/".join(partial_res_save)
                with gzip.open(partial_res_save, "wb") as f:
                    pickle.dump({"fail_pred_raw1s": fail_pred_raw1s, 
                                 "fail_pred_raw2s": fail_pred_raw2s, 
                                 "fail_pred_labels": fail_pred_labels,
                                 "adv_raw1s": adv_raw1s,
                                 "adv_raw2s": adv_raw2s,
                                 "adv_labels": adv_labels}, f)
        if res_save is not None:
            print ("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_raw1s), len(fail_pred_raw1s)))
            with gzip.open(res_save, "wb") as f:
                pickle.dump({"fail_pred_raw1s": fail_pred_raw1s, 
                             "fail_pred_raw2s": fail_pred_raw2s, 
                             "fail_pred_labels": fail_pred_labels,
                             "adv_raw1s": adv_raw1s,
                             "adv_raw2s": adv_raw2s,
                             "adv_labels": adv_labels}, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/n_total))

if __name__ == "__main__":

    from gensim.models.word2vec import Word2Vec
    import numpy as np
    
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
    
    oj = OJ104(path="../data_clone/oj.pkl.gz")
    training_set = oj.train
    valid_set = oj.dev
    test_set = oj.test
    with gzip.open('../data_clone/oj_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_clone/oj_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    
    model = BatchProgramCC(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                           USE_GPU, embeddings).cuda()
    model.load_state_dict(torch.load('../model_clone/astnn/model.pt'))
    
    #atk = AdversarialTrainingAttacker(oj, symtab, model, word2vec)
    #atk.attack_all(40, 50, 1, "../model_clone/astnn/uid_rename_atk_cand40_iter50_relax100.advsamples.pkl.gz", adv_sample_size=5000)

    atk = AdversarialTrainingInsAttacker(oj, instab, model, word2vec)
    atk.attack_all(40, 20, 1, "../model_clone/astnn/stmt_insert_atk_cand40_iter20_relax100.advsamples.pkl.gz", adv_sample_size=5000)