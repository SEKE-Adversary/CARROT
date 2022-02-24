# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:38:44 2020

@author: 63561
"""

import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class Pipeline:
    def __init__(self, root):
        self.root = root
        self.sources = None
        self.blocks = None
        self.train_pairs = None
        self.dev_pairs = None
        self.test_pairs = None
        self.train_file_path = '../data_clone/train/pair.pkl'
        self.dev_file_path = '../data_clone/dev/pair.pkl'
        self.test_file_path = '../data_clone/test/pair.pkl'
        self.size = None

    # parse source code
    def parse_source(self, output_file, option):
        path = self.root+'/'+output_file
        if os.path.exists(path) and option == 'existing':
            sources = pd.read_pickle(path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            # train, dev programs are the same
            program_paths = ["../data_clone/train/program.pkl",
                             "../data_clone/test/program.pkl"]
            source_list = []
            for program_path in program_paths:
                source = pd.read_pickle(program_path)
                source_list.append(source)
            sources = pd.concat(source_list)
            # sources = sources[[not i for i in sources['code'].duplicated()]]
            sources['code'] = sources['code'].apply(parser.parse)
            sources.to_pickle(path)
        self.sources = sources
        print("Total program numbers:", len(self.sources))
        return sources

    # create clone pairs
    def read_pairs(self):
        self.train_pairs = pd.read_pickle(self.train_file_path)
        self.dev_pairs = pd.read_pickle(self.dev_file_path)
        self.test_pairs = pd.read_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        data_path = self.root
        if not input_file:
            input_file = self.train_file_path
        pairs = pd.read_pickle(input_file)
        train_ids = pairs['id1'].append(pairs['id2']).unique()

        trees = self.sources.set_index('id',drop=False).loc[train_ids]
        if not os.path.exists(data_path+'train/embedding'):
            os.mkdir(data_path+'train/embedding')
        sys.path.append('../preprocess-astnn/')
        from prepare_data import get_sequences as func

        def trans_to_sequences(ast):
            sequence = []
            func(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        # trees.to_csv(data_path+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, max_final_vocab=3000)
        w2v.save(data_path+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'/train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.DataFrame(self.sources, copy=True)
        trees['code'] = trees['code'].apply(trans2seq)
        if 'label' in trees.columns:
            trees.drop('label', axis=1, inplace=True)
        self.blocks = trees

    # merge pairs
    def merge(self,data_path,part):
        pairs = pd.read_pickle(data_path)
        pairs['id1'] = pairs['id1'].astype(int)
        pairs['id2'] = pairs['id2'].astype(int)
        df = pd.merge(pairs, self.blocks, how='left', left_on='id1', right_on='id')
        df = pd.merge(df, self.blocks, how='left', left_on='id2', right_on='id')
        df.drop(['id_x', 'id_y'], axis=1,inplace=True)
        df.dropna(inplace=True)

        df.to_pickle(self.root+'/'+part+'/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('read id pairs...')
        self.read_pairs()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs()
        print('merge pairs and blocks...')
        self.merge(self.train_file_path, 'train')
        self.merge(self.dev_file_path, 'dev')
        self.merge(self.test_file_path, 'test')
#         print(len(self.blocks))

if __name__ == "__main__":
    
    ppl = Pipeline('../data_clone/')
    ppl.run()