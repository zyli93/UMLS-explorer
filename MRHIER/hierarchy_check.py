import argparse
import pandas as pd
import networkx as nx

# This script:
# 1. checks if each context of each source vocabulary is a tree structure
# 2. dumps MRHIER metadata in this format: SAB, # of AUIs, # of contexts, # of trees

parser = argparse.ArgumentParser(description='MRHIER trees')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print cycles in context/source')
parser.add_argument('-s', '--summary', action='store_true', default=False, help='print metadata')
opt = parser.parse_args()

mrhier_filled = pd.read_csv('MRHIER_filled.csv', sep='|', dtype=object)
sources = mrhier_filled['SAB'].unique()

if opt.summary:
    print("SAB,#AUIs,#CXNs,#trees")

# Separate each source vocabulary
for sab in sources:
    source = mrhier_filled[mrhier_filled['SAB'] == sab]
    contexts = source['CXN'].unique()
    tree_count = 0
    if opt.summary:
        print(sab, end=',')
        print(source['AUI'].unique().size, end=',')
        print(contexts.size, end=',')
    else:
        print("Source Vocabulary: {}".format(sab))
        print("Contexts: {}".format(contexts))
    # Separate each context
    for cxn in contexts:
        cxn_source = source[source['CXN'] == cxn]
        # Build the graph
        G = nx.Graph()
        for index, row in cxn_source.iterrows():
            if pd.isna(row['PTR']):
                continue
            parents = row['PTR'].split('.')
            for i, p in enumerate(parents[:-1]):
                G.add_edge(p, parents[i+1])
            G.add_edge(parents[-1], row['AUI'])
        if not opt.summary:
            print("context {}".format(cxn))
            if nx.is_tree(G):
                print("is tree: True")
            else:
                print("is tree: False")
                if opt.verbose:
                    print("cycles if any: {}".format(nx.find_cycle(G)))
        elif nx.is_tree(G):
            tree_count += 1

    if opt.summary:
        print(tree_count)