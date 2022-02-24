# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:26:02 2020

@author: DrLC
"""

from dataset import CodeChef
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUClassifier, GRUEncoder
from modifier import TokenModifier, InsModifier

import time
import torch
import argparse
import pickle, gzip
import os, sys
from sklearn import metrics

class Attacker(object):
    
    def __init__(self, dataset, symtab, classifier):
        
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
    
    def attack(self, x, y, uids, n_candidate=100, n_iter=20):
        
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print ("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
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
                new_x, new_uid_cand = self.tokenM.rename_uid(x, y, uids[k], k, n_candidate)
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).cuda().permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], [p.cpu().numpy()]
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                           y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, x, y
    
    def attack_all(self, n_candidate=100, n_iter=20, dump_samples_path=None):
        
        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['x'], b['y'], self.syms['te'][b['id'][0]], n_candidate, n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y[0]))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                sample_dict["adv_x"].append(adv_x[0])
                sample_dict["adv_y"].append(adv_y[0])
            else:
                sample_dict["adv_x"].append(None)
                sample_dict["adv_y"].append(-1)
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
        if dump_samples_path != None:
            with gzip.open(dump_samples_path, "wb") as f:
                pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class InsAttacker(object):
    
    def __init__(self, dataset, instab, classifier):
        
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None) # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab
    
    # only support single x: a token-idx list
    def attack(self, x, y, poses, n_candidate=100, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print ("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_del, new_insertDict_del = self.insM.remove(x[0], n_candidate_del)
            new_x_add, new_insertDict_add = self.insM.insert(x[0], n_candidate_ins)
            new_x = new_x_del + new_x_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            feed_new_x = [_x[:self.cl.max_len] for _x in new_x] # this step is important
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.cuda().permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                    return True, [_x], [p.cpu().numpy()]

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                            y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= len(new_x):    # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print ("FAIL!")
        return False, x, y
    
    def attack_all(self, n_candidate=100, n_iter=20, dump_samples_path=None):
        
        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['x'], b['y'], self.inss['stmt_te'][b['id'][0]], n_candidate, n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y[0]))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                sample_dict["adv_x"].append(adv_x[0])
                sample_dict["adv_y"].append(adv_y[0])
            else:
                sample_dict["adv_x"].append(None)
                sample_dict["adv_y"].append(-1)
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
        if dump_samples_path != None:
            with gzip.open(dump_samples_path, "wb") as f:
                pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class AttackerRandom(object):
    
    def __init__(self, dataset, symtab, classifier):
        
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.cl = classifier
        self.d = dataset
        self.syms = symtab
    
    def attack(self, x, y, uids, n_iter=20):
        
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print ("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
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
                new_x, new_uid_cand = self.tokenM.rename_uid_random(x, uids[k], k)
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).cuda().permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print ("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                               (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], [p.cpu().numpy()]
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print ("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                           (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                           y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print ("rej\t%s" % k)
        print ("FAIL!")
        return False, x, y
    
    def attack_all(self, n_iter=20, dump_samples_path=None):
        
        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['x'], b['y'], self.syms['te'][b['id'][0]], n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y[0]))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                sample_dict["adv_x"].append(adv_x[0])
                sample_dict["adv_y"].append(adv_y[0])
            else:
                sample_dict["adv_x"].append(None)
                sample_dict["adv_y"].append(-1)
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
        if dump_samples_path != None:
            with gzip.open(dump_samples_path, "wb") as f:
                pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

class InsAttackerRandom(object):
    
    def __init__(self, dataset, instab, classifier):
        
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None) # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab
    
    # only support single x: a token-idx list
    def attack(self, x, y, poses, n_iter=20):
        
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).cuda().permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print ("SUCC! Original mistake.")
            return True, x, [torch.argmax(old_prob).cpu().numpy()]
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            iter += 1

            # get insertion candidates
            new_x, new_insertDict = self.insM.insert_remove_random(x[0])
            if new_x == []: # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            feed_new_x = [_x[:self.cl.max_len] for _x in new_x] # this step is important
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.cuda().permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    print ("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                            (self.insM.insertDict["count"], insD["count"], 
                                y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                    return True, [_x], [p.cpu().numpy()]

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:
                print ("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                        (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"], 
                            y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                self.insM.insertDict = new_insertDict[new_prob_idx] # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
                print ("rej\t%s" % "")
            if n_stop >= 10:
                iter = n_iter
                break
        print ("FAIL!")
        return False, x, y
    
    def attack_all(self, n_iter=20, dump_samples_path=None):
        
        sample_dict = {"x": [], "y": [], "adv_x": [], "adv_y": []}
        n_succ = 0
        total_time = 0
        trues, preds = [], []
        st_time = time.time()
        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            sample_dict["x"].append(b['x'][0])
            sample_dict["y"].append(b['y'][0])
            print ("\t%d/%d\tID = %d\tY = %d" % (i+1, self.d.test.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, adv_x, adv_y = self.attack(b['x'], b['y'], self.inss['stmt_te'][b['id'][0]], n_iter)
            trues.append(int(b['y'][0]))
            preds.append(int(adv_y[0]))
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                sample_dict["adv_x"].append(adv_x[0])
                sample_dict["adv_y"].append(adv_y[0])
            else:
                sample_dict["adv_x"].append(None)
                sample_dict["adv_y"].append(-1)
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
        if dump_samples_path != None:
            with gzip.open(dump_samples_path, "wb") as f:
                pickle.dump(sample_dict, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time()-st_time, n_succ/self.d.test.get_size()))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', required=True)
    parser.add_argument('-attn', action='store_true')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda")
    
    vocab_size = 3000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 4
    max_len = 300

    cc = CodeChef('../data_defect/codechef.pkl.gz',
                  max_len=max_len,
                  vocab_size=vocab_size)
    training_set = cc.train
    valid_set = cc.dev
    test_set = cc.test
    with gzip.open('../data_defect/codechef_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../data_defect/codechef_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    
    '''
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc,
                                hidden_size, num_classes, max_len, attn=opt.attn).cuda()
    classifier.load_state_dict(torch.load('../model_defect/lstm/model-13.pt'))

    #atk = Attacker(cc, symtab, classifier)
    #atk.attack_all(40, 50)

    #atk = InsAttacker(cc, instab, classifier)
    #atk.attack_all(40, 20)

    #atk = AttackerRandom(cc, symtab, classifier)
    #atk.attack_all(100)

    atk = InsAttackerRandom(cc, instab, classifier)
    atk.attack_all(20)
    '''

    enc = GRUEncoder(embedding_size, hidden_size, n_layers)
    classifier = GRUClassifier(vocab_size, embedding_size, enc,
                               hidden_size, num_classes, max_len, attn=opt.attn).cuda()
    classifier.load_state_dict(torch.load('../model_defect/gru/model-15.pt'))

    #atk = Attacker(cc, symtab, classifier)
    #atk.attack_all(40, 50)

    #atk = InsAttacker(cc, instab, classifier)
    #atk.attack_all(40, 20)

    #atk = AttackerRandom(cc, symtab, classifier)
    #atk.attack_all(100)

    atk = InsAttackerRandom(cc, instab, classifier)
    atk.attack_all(20)
    