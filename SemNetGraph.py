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
top_struct = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTR', dtype = 'unicode', delimiter = '|')
inherited_struct = np.genfromtxt('/Users/qinyilong/Desktop/ScAi/SRSTRE2', dtype = 'unicode', delimiter = '|')

# Delete the last columns because they are empty
top_struct = np.delete(top_struct, 4, axis = 1)
inherited_struct = np.delete(inherited_struct, 3, axis = 1)

print("Shape of SRSTR (top node structure): " + str(top_struct.shape))
print("Shape of SRSTRE2 (inheritance structure): " + str(inherited_struct.shape))

# Partition top node structure according to entries' link status:
# D = Defined for the Arguments and its children;
# B = Blocked;
# DNI = Defined but Not Inherited by the children of the Arguments

top_struct_d = top_struct[top_struct[:, 3] == 'D', :]
top_struct_b = top_struct[top_struct[:, 3] == 'B', :]
top_struct_dni = top_struct[top_struct[:, 3] == 'DNI', :]

print("Shapes of top node relationships: ")
print("Defined: " + str(top_struct_d.shape))
print("Blocked: " + str(top_struct_b.shape))
print("Defined but Not Inherited: " + str(top_struct_dni.shape))

top = nx.Graph()
for entry in top_struct_d:
    top.add_node(entry[0])
    top.add_node(entry[2])
    top.add_edge(entry[0], entry[2], rel = entry[1])

print(top.number_of_nodes())

options = {
    'node_color': 'red',
    'node_size': 5,
}
nx.draw(top, **options, with_labels=True)
plt.show()