# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:01:34 2021

@author: DrLC
"""

import pycparser
import torch

import dgl

class SingleNode():
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token


def extract_ast(root, idx, nodes, begins, ends):

    for s, c in root.children():
        assert isinstance(s, str)
        assert isinstance(c, pycparser.c_ast.Node)
        name = None
        if hasattr(c, "name"):
            assert isinstance(c.name, str) or c.name is None
            name = c.name
        elif hasattr(c, "declname"):
            assert isinstance(c.declname, str) or c.declname is None
            name = c.declname
        elif hasattr(c, "names"):
            name = " ".join(c.names)
        if name:
            nodes.append(SingleNode(c).get_token())
        else:
            nodes.append(SingleNode(c).get_token())
        begins.append(len(nodes)-1)
        ends.append(idx)
        # print (nodes[-1], begins[-1], ends[-1])
        nodes, begins, ends = extract_ast(c, begins[-1], nodes, begins, ends)
    return nodes, begins, ends
    
def build_dgl_graph(nodes, begins, ends):
    
    b = torch.tensor(begins)
    e = torch.tensor(ends)
    n = torch.tensor(nodes)
    g = dgl.graph((b, e))
    g.ndata['type'] = n
    return g

if __name__ == "__main__":
    
    import networkx as nx
    
    parser = pycparser.c_parser.CParser()
    
    ast = parser.parse("float foo(int, float); float foo(int arg, float arg2) {return (float)arg;}")
    
    n, b, e = extract_ast(ast, 0, [SingleNode(ast).get_token()], [], [])
    n_idx = [n.index(i) for i in n]
    
    g = build_dgl_graph(n_idx, b, e)
    
    nx_g = g.to_networkx()
    labels = {i: n[i] for i in range(len(n))}
    pos = nx.drawing.nx_pydot.graphviz_layout(nx_g, prog="dot")
    nx.draw(nx_g, pos, labels=labels, with_labels=True, node_color=['#ffffff' for _ in labels])
