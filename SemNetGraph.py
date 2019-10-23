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

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(graph):
    # visualize_graph visualizes a graph using a set of options found through trial and error
    # networkX takes options for node/edge size/color among other things
    # In order for each node/edge to have its own size/color, create a list of all nodes' sizes/colors

    # Create a list of node sizes based on frequency of appearance
    node_sizes = [styFreq[n] * 10 for n in graph.nodes]
    # Create a list of edge colors based on its relation type
    edge_colors = [attr['color'] for n1, n2, attr in graph.edges(data=True)]
    # Use graphviz_layout to get cleaner layout
    positions = nx.nx_agraph.graphviz_layout(graph, prog='neato')

    options = {
        'pos': positions,           # layout option
        'node_color': 'khaki',
        'node_size': node_sizes,
        'with_labels': True,
        'edge_color': edge_colors,
        'font_size': 5,
    }

    plt.figure()
    nx.draw(graph, **options)
    plt.show()


# import files of interest
srstr = np.genfromtxt('/Users/zyli/Research/UMLS-explorer/NET/SRSTR', dtype='unicode', delimiter='|')
srstre2 = np.genfromtxt('/Users/zyli/Research/UMLS-explorer/NET/SRSTRE2', dtype='unicode', delimiter='|')

# srstr = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTR', dtype='unicode', delimiter='|')
# srstre2 = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTRE2', dtype='unicode', delimiter='|')

# Delete the last columns because they are empty
srstr = np.delete(srstr, 4, axis=1)
srstre2 = np.delete(srstre2, 3, axis=1)

print("Shape of SRSTR file (distance-1): " + str(srstr.shape))
print("Shape of SRSTRE2 file (fully inherited): " + str(srstre2.shape))

# Split the data into STY & REL
srstr_sty = srstr[:543, :]
srstr_rel = srstr[543:, :]
srstre2_sty = srstre2[:6105, :]
srstre2_rel = srstre2[6105:, :]

# Partition distance-1 structure according to entries' link status:
# D = Defined for the Arguments and its children;
# B = Blocked;
# DNI = Defined but Not Inherited by the children of the Arguments

srstr_d = srstr_sty[srstr_sty[:, 3] == 'D', :]
srstr_b = srstr_sty[srstr_sty[:, 3] == 'B', :]
srstr_dni = srstr_sty[srstr_sty[:, 3] == 'DNI', :]

print("Shapes of SRSTR relationships among STYs: ")
print("Defined: " + str(srstr_d.shape))
print("Blocked: " + str(srstr_b.shape))
print("Defined but Not Inherited: " + str(srstr_dni.shape))

# Create multi-directed-graphs for SRSTR and SRSTRE2
srstr_graph = nx.MultiDiGraph(name='SRSTR defined relationship graph')
srstr_trees = nx.MultiDiGraph(name='SRSTR defined tree (only isa relation)')
for entry in srstr_d:
    if entry[2] != '':  # Disconnect 4 topmost nodes from ''
        if entry[1] == 'isa':
            srstr_graph.add_edge(entry[0], entry[2], color='lightcoral', relation=entry[1])
            srstr_trees.add_edge(entry[0], entry[2], color='lightcoral', relation=entry[1])
        else:
            srstr_graph.add_edge(entry[0], entry[2], color='cyan', relation=entry[1])

srstre2_graph = nx.MultiDiGraph(name='SRSTRE2 relationship graph')
for entry in srstre2_sty:
    if entry[2] != '':  # Disconnect 4 topmost nodes from ''
        if entry[1] == 'isa':
            srstre2_graph.add_edge(entry[0], entry[2], color='lightcoral', relation=entry[1])
        else:
            srstre2_graph.add_edge(entry[0], entry[2], color='cyan', relation=entry[1])

# Print number of nodes and edges
# We can see number of nodes present in SRSTR and SRSTRE2 are the same.
print(nx.info(srstr_graph))
print(nx.info(srstre2_graph))

# SRSTR ans SRSTRE2 have the same nodes, which make up the entirety of Semantic Types
assert (list(srstr_graph.nodes).sort() == list(srstre2_graph.nodes).sort())

# STY frequency in SRSTR
styFreq = dict((el, 0) for el in list(srstr_graph.nodes))
for entry in srstr_d:
    if entry[2] != '':
        styFreq[entry[0]] += 1
        styFreq[entry[2]] += 1

# REL frequency in SRSTRE2
relFreq_srstre2 = {}
for rel in srstre2_sty[:, 1]:
    if rel in relFreq_srstre2:
        relFreq_srstre2[rel] += 1
    else:
        relFreq_srstre2[rel] = 1

# with open('rel_freq_SRSTRE2.pickle', 'wb') as handle:
#     pickle.dump(relFreq_srstre2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Reflexive and symmetric relationships
reflex_count = 0
symmetric_count = 0
for entry in srstr_d:
    if entry[0] == entry[2]:
        # print("Reflexive relationship: ")
        # print(entry)
        reflex_count += 1
    else:
        for anotherEntry in srstr_d:
            if anotherEntry[0] == entry[2] and anotherEntry[2] == entry[0]:
                # print("Symmetric relationship: ")
                # print(entry)
                # print(anotherEntry)
                symmetric_count += 1

print("Total number of reflexive relationships (xRx): " + str(reflex_count))
print("Total number of symmetric relationships (xRy yRx): " + str(symmetric_count))

visualize_graph(srstr_graph)
visualize_graph(srstr_trees)