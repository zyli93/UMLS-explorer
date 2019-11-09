import pickle
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from plotly.offline import plot

# import SRSTR
srstr = pd.read_csv('/Users/qinyilong/Desktop/ScAi/SRSTR', sep='|', header=None, dtype=object)
srstr = srstr.drop(4, axis=1)     # delete the empty last column

# import semantic type frequencies
with open('semanticTypes.pickle', 'rb') as handle:
    sty = pickle.load(handle)

# Extract defined relations among semantic types
srstr_sty = srstr.loc[:542]
srstr_sty.columns = ["STY1", "RL", "STY2", "LS"]
srstr_sty_d = srstr_sty[srstr_sty.LS == "D"]
srstr_sty_b = srstr_sty[srstr_sty.LS == "B"]
srstr_sty_dni = srstr_sty[srstr_sty.LS == "DNI"]

# Create multi-directed-graphs with full SRSTR relations and only isa relations
srstr_trees = nx.MultiDiGraph(name='SRSTR defined tree (only isa relation)')
for index, row in srstr_sty_d.iterrows():
    if pd.notna(row['STY2']):  # Disconnect topmost nodes from ''
        if row['RL'] == 'isa':
            srstr_trees.add_edge(row['STY1'], row['STY2'], relation=row['RL'])

color_dict = dict((el, None) for el in sty)
current_level = ['Event']
current_color = [255, 0, 0, 1.0]
while len(current_level) != 0:
    next_level = []
    for sty in current_level:
        color_dict[sty] = 'rgba' + str(tuple(current_color))
        next_level += list(srstr_trees.predecessors(sty))
    current_color[1] += 45
    current_level = next_level

current_level = ['Entity']
current_color = [0, 0, 255, 1.0]  # rgba
while len(current_level) != 0:
    next_level = []
    for sty in current_level:
        color_dict[sty] = 'rgba' + str(tuple(current_color))
        next_level += list(srstr_trees.predecessors(sty))
    current_color[1] += 45
    current_level = next_level

with open('tree_color.pickle', 'wb') as handle:
    pickle.dump(color_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

position = nx.nx_agraph.graphviz_layout(srstr_trees, prog='neato')

edge_x = []
edge_y = []
for node1, node2, attribute in srstr_trees.edges(data=True):
    x0, y0 = position[node1]
    x1, y1 = position[node2]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
semantic_types = []
node_color = []
for node in srstr_trees.nodes:
    x, y = position[node]
    node_x.append(x)
    node_y.append(y)
    semantic_types.append(node)
    node_color.append(color_dict[node])

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    text=semantic_types,
    hoverinfo='text',
    marker=dict(
        color=node_color,
        size=10,
        line_width=2
    )
)

axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

layout = go.Layout(
    title='Semantic Network trees',
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
    )
)

fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
plot(fig, filename='semnet_color_trees.html')