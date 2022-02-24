# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:21:54 2021

@author: DrLC
"""

import pycparser

def const_norm(val):
    
    if val[0] == '\'' and val[-1] == '\'':
        return "<char>"
    elif val[0] == '"' and val[-1] == '"':
        return "<str>"
    try:
        int(val)
        return "<int>"
    except ValueError:
        try:
            float(val)
            return "<fp>"
        except ValueError:
            return val

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
        return const_norm(token)


def extract_ast(root, idx, nodes, begins, ends):

    for s, c in root.children():
        nodes.append(SingleNode(c).get_token())
        begins.append(len(nodes)-1)
        ends.append(idx)
        # print (nodes[-1], begins[-1], ends[-1])
        nodes, begins, ends = extract_ast(c, begins[-1], nodes, begins, ends)
    return nodes, begins, ends

if __name__ == "__main__":
    
    parser = pycparser.c_parser.CParser()
    
    ast = parser.parse("float foo(int, float); float foo(int arg, float arg2) {return (float)(arg+arg2)-'0'*100/0.3;}")
    
    n, b, e = extract_ast(ast, 0, [SingleNode(ast).getn_token()], [], [])