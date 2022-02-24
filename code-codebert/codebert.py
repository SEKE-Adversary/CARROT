# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 21:42:56 2021

@author: DrLC
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging

import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer

logger = logging.getLogger(__name__)

class CodeBERTClassifier(nn.Module):
    
    def __init__(self, model_path, num_labels, device=None):
        
        super(CodeBERTClassifier, self).__init__()
        
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        self.block_size = 512
        self.embed = self.model.roberta.embeddings.word_embeddings
        self.vocab_size = self.embed.weight.size()[0]
        self.x_size = self.embed.weight.size()[-1]
        self.device = device

    def tokenize(self, inputs, cut_and_pad=False, ret_id=False):
        
        rets = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for sent in inputs:
            if cut_and_pad:
                tokens = self.tokenizer.tokenize(sent)[:self.block_size-2]
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                padding_length = self.block_size - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_length
            else:
                tokens = self.tokenizer.tokenize(sent)
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets

    def run_batch(self, inputs_src, labels=None):
        
        inputs = self.tokenize(inputs_src, cut_and_pad=True, ret_id=True)
        inputs = torch.tensor(inputs, dtype=torch.long).to(self.device)
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        outputs = self.model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), labels=labels)
        
        if labels is not None:
            loss, logits = outputs.loss, outputs.logits
            return logits, loss
        else:
            return outputs[0]

    def forward(self, inputs, labels=None):
        
        return self.run_batch(inputs, labels)

    def prob(self, inputs):

        logits = self.forward(inputs)
        prob = nn.Softmax(dim=-1)(logits)
        return prob

    def grad(self, inputs, labels):
        
        self.zero_grad()
        self.embed.weight.retain_grad() # (50265, 768)
        #print(self.vocab_size, self.x_size)

        logits, loss = self.forward(inputs, labels)
        loss.backward()
        #print(self.embed.weight.grad)

        return self.embed.weight.grad

if __name__ == "__main__":
    
    from dataset import OJ104
    from torch import optim
    
    oj = OJ104(path="../data/oj.pkl.gz")
    model = CodeBERTClassifier('../model_codebert_base_mlm', 104).train()
    opt = optim.Adam(model.parameters(), lr=1e-5)
    
    opt.zero_grad()
    
    b = oj.train.next_batch(3)
    logits, loss = model.run_batch(b['x'], b['y'])
    
    loss.backward()
    opt.step()
