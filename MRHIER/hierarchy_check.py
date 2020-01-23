import pandas as pd
import networkx as nx

# This script checks if each context of each source vocabulary is a tree structure

mrhier_filled = pd.read_csv('MRHIER_filled.csv', sep='|', dtype=object)
sources = mrhier_filled['SAB'].unique()

# Separate each source vocabulary
for sab in sources:
    source = mrhier_filled[mrhier_filled['SAB'] == sab]
    print("Source Vocabulary: {}".format(sab))
    contexts = source['CXN'].unique()
    print("Contexts: {}".format(contexts))
    # Separate each context
    for cxn in contexts:
        print("context {}".format(cxn))
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
        print("is tree: {}".format(nx.is_tree(G)))
        # verbose output
        # print("cycles if any: {}".format(nx.find_cycle(G)))