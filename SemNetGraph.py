# Semantic Network Visualization

# basic tables:
# - SRDEF: basic info about the Semantic Types and Relations
# - SRSTR: stucture of the Network
#
# ancillary tables: fully inherited set of relations
# - SRSTRE1: UI's
# - SRSTRE2: names
#
# bookkeeping tables:
# - SRFIL: description of each table
# - SRFLD: description of each field

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# import files of interest
srstr = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTR', dtype = 'unicode', delimiter = '|')
srstre2 = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTRE2', dtype = 'unicode', delimiter = '|')

# Delete the last columns because they are empty
srstr = np.delete(srstr, 4, axis = 1)
srstre2 = np.delete(srstre2, 3, axis = 1)

print("Shape of SRSTR (top node structure): " + str(srstr.shape))
print("Shape of SRSTRE2 (inheritance structure): " + str(srstre2.shape))

# Partition top node structure according to entries' link status:
# D = Defined for the Arguments and its children;
# B = Blocked;
# DNI = Defined but Not Inherited by the children of the Arguments

srstr_d = srstr[srstr[:, 3] == 'D', :]
srstr_b = srstr[srstr[:, 3] == 'B', :]
srstr_dni = srstr[srstr[:, 3] == 'DNI', :]

print("Shapes of top node relationships: ")
print("Defined: " + str(srstr_d.shape))
print("Blocked: " + str(srstr_b.shape))
print("Defined but Not Inherited: " + str(srstr_dni.shape))

# Create directed graphs for SRSTR and SRSTRE2
srstr_graph = nx.DiGraph()
for entry in srstr:
    # Disconnect 4 topmost nodes from ''
    if entry[2] == '':
        continue
    else:
        srstr_graph.add_node(entry[0])
        srstr_graph.add_node(entry[2])
        srstr_graph.add_edge(entry[0], entry[2], relation = entry[1])

srstre2_graph = nx.DiGraph()
for entry in srstre2:
    srstre2_graph.add_node(entry[0])
    srstre2_graph.add_node(entry[2])
    srstre2_graph.add_edge(entry[0], entry[2], relation = entry[1])

# Print number of nodes and edges
# We can see number of nodes present in SRSTR and SRSTRE2 are the same.
print("Number of nodes for SRSTR: " + str(srstr_graph.number_of_nodes()))
print("Number of edges for SRSTR: " + str(srstr_graph.number_of_edges()))
print("Number of nodes for SRSTRE2: " + str(srstre2_graph.number_of_nodes()))
print("Number of edges for SRSTRE2: " + str(srstre2_graph.number_of_edges()))

# In case lists of nodes and edges are needed
srstr_nodes = list(srstr_graph.nodes)
srstr_edges = list(srstr_graph.edges)
srstre2_nodes = list(srstre2_graph.nodes)
srstre2_edges = list(srstre2_graph.edges)

# options = {
#     'node_color': 'red',
#     'node_size': 5,
#     'with_labels': True,
#     'alpha': 0.5,
#     'edge_color': 'blue',
# }
#
# nx.draw(srstr_graph, **options)
# plt.show()
