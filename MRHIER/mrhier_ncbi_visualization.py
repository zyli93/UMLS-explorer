import pickle
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from plotly.offline import plot

print('loading data')

mrhier = pd.read_csv('/Users/qinyilong/Desktop/ScAi/MRHIER.RRF', sep='|', header=None, dtype=object)
mrhier = mrhier.drop(9, axis=1)
mrhier.columns = ['CUI', 'AUI', 'CXN', 'PAUI', 'SAB', 'RELA', 'PTR', 'HCD', 'CVF']
mrhier_ncbi = mrhier[mrhier['SAB'] == 'NCBI']
mrhier_ncbi = mrhier_ncbi[mrhier_ncbi['PTR'].notna()]

mrconso = pd.read_csv('/Users/qinyilong/Desktop/ScAi/MRCONSO.RRF', sep='|', header=None, dtype=object)
mrconso = mrconso.drop(18, axis=1)
mrconso.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]

with open('CUI_to_STYs.pickle', 'rb') as handle:
    cui_to_stys = pickle.load(handle)

print('creating graph')

G = nx.DiGraph(name='MRHIER NCBI hierarchy graph')
for index, row in mrhier_ncbi.iterrows():
    parents = row['PTR'].split('.')
    G.add_edges_from(zip(parents[:-1], parents[1:]))
    G.add_edge(parents[-1], row['AUI'])

print(nx.is_tree(G))

print('laying out')

position = nx.nx_agraph.graphviz_layout(G, prog='neato')

print('saving coordinates for plotly')

edge_x = []
edge_y = []
for edge in G.edges:
    x0, y0 = position[edge[0]]
    x1, y1 = position[edge[1]]
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
node_text = []
for node in G.nodes:
    x, y = position[node]
    node_x.append(x)
    node_y.append(y)
    cui = mrconso[mrconso.AUI == node].CUI.iloc[0]
    sty = cui_to_stys[cui]
    node_text.append(node + '\n' + cui + '\n' + sty)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    text=auis,
    hoverinfo='text',
    marker=dict(
        color='black',
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
    title='MRHIER NCBI tree',
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
    )
)

print('plotly ploting')

fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
plot(fig, filename='mrhier_ncbi_visualization.html')