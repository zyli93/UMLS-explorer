This file provides instructions to visualize relationships among semantic types presented in UMLS Semantic Network.

Content of relational format files (from UMLS reference manual):
basic tables:
- SRDEF: basic info about the Semantic Types and Relations
- SRSTR: stucture of the Network
ancillary tables: fully inherited set of relations
- SRSTRE1: UI's
- SRSTRE2: names
bookkeeping tables:
- SRFIL: description of each table
- SRFLD: description of each field

When visualizing relationships, we use SRSTR.
1. Load data from SRSTR
2. Split data
    - semantic types and relationships
    - relationships that are defined, blocked, defined but not inherited
3. Create MultiDiGraph objects: srstr_graph, srstr_trees
    - srstr_graph includes all relationships
    - srstr_trees includes all isa relationships
    Note: 
    - when adding edges, attribute 'color' is different for 'isa' and associative relationships
4. Create a dictionary of frequency of appearance for semantic types
5. Create graph with nx.draw(graph, **options)
    Notes: important options
    - 'pos': nx.nx_agraph.graphviz_layout(graph, prog='neato')
        pygraphviz has cleaner layout compared to networkx's
        prog parameter: https://stackoverflow.com/questions/21978487/improving-python-networkx-graph-layout
    - 'node_size': a list of node sizes correspond to their frequencies
        The bigger the node, the more connection it has
    - 'edge_color': a list of edge colors created from attribute 'color' of edges