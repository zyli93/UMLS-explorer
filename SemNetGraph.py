import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# import SRSTR
srstr = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTR', dtype='unicode', delimiter='|')
srstr = np.delete(srstr, 4, axis=1)     # delete the empty last column
# import semantic type frequencies
with open('semanticTypes.pickle', 'rb') as handle:
    sty = pickle.load(handle)

# Extract defined relations among semantic types
srstr_sty = srstr[:543, :]
srstr_d = srstr_sty[srstr_sty[:, 3] == 'D', :]

styFreq = dict((el, 0) for el in sty)
# Create multi-directed-graphs with full SRSTR relations and only isa relations
srstr_graph = nx.MultiDiGraph(name='SRSTR defined relationship graph')
srstr_trees = nx.MultiDiGraph(name='SRSTR defined tree (only isa relation)')
for entry in srstr_d:
    if entry[2] != '':  # Disconnect topmost nodes from ''
        styFreq[entry[0]] += 1
        styFreq[entry[2]] += 1
        if entry[1] == 'isa':
            srstr_graph.add_edge(entry[0], entry[2], color='lightcoral', relation=entry[1])
            srstr_trees.add_edge(entry[0], entry[2], color='lightcoral', relation=entry[1])
        else:
            srstr_graph.add_edge(entry[0], entry[2], color='cyan', relation=entry[1])

# Choose graph and node_sizes
graph = srstr_graph
node_sizes = [styFreq[n] * 10 for n in srstr_graph.nodes]  # node sizes based on frequency of appearance

options = {
    # Use graphviz_layout to get cleaner layout
    'pos': nx.nx_agraph.graphviz_layout(srstr_trees, prog='neato'),
    'node_color': 'khaki',
    'node_size': node_sizes,
    'with_labels': True,
    # Create a list of edge colors based on its relation type
    'edge_color': [attr['color'] for n1, n2, attr in graph.edges(data=True)],
    'font_size': 10,
}

nx.draw(graph, **options)
plt.show()