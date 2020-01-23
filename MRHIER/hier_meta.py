import pandas as pd
import networkx as nx

# This script dumps MRHIER metadata in the following format
# SAB, # of AUIs, # of contexts, # of trees

print("SAB,#AUIs,#CXNs,#trees")

mrhier_filled = pd.read_csv('MRHIER_filled.csv', sep='|', dtype=object)
sources = mrhier_filled['SAB'].unique()

# Separate each source vocabulary
for sab in sources:
    source = mrhier_filled[mrhier_filled['SAB'] == sab]
    print(sab, end=',')
    print(source['AUI'].unique().size, end=',')
    contexts = source['CXN'].unique()
    tree_count = 0
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
        # Check tree structure
        if nx.is_tree(G):
            tree_count += 1
    print(contexts.size, end=',')
    print(tree_count)