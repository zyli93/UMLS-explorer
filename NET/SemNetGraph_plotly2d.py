import pickle
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import streamlit as st

st.sidebar.title('UMLS Semantic Network Visualization')

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

sty_freq = dict((el, 0) for el in sty)
# Create multi-directed-graphs with full SRSTR relations and only isa relations
srstr_graph = nx.MultiDiGraph(name='SRSTR defined relationship graph')
srstr_trees = nx.MultiDiGraph(name='SRSTR defined tree (only isa relation)')
for index, row in srstr_sty_d.iterrows():
    if pd.notna(row['STY2']):  # Disconnect topmost nodes from ''
        sty_freq[row['STY1']] += 1
        sty_freq[row['STY2']] += 1
        if row['RL'] == 'isa':
            srstr_graph.add_edge(row['STY1'], row['STY2'], relation=row['RL'])
            srstr_trees.add_edge(row['STY1'], row['STY2'], relation=row['RL'])
        else:
            srstr_graph.add_edge(row['STY1'], row['STY2'], relation=row['RL'])

assert(len(list(srstr_trees.nodes)) == len((list(srstr_graph.nodes))))

layout_option = st.sidebar.radio(
    'Layout options',
    ('Hierarchical', 'Hierarchical + Associative')
)

if layout_option == 'Hierarchical':
    position = nx.nx_agraph.graphviz_layout(srstr_trees, prog='neato')
elif layout_option == 'Hierarchical + Associative':
    position = nx.nx_agraph.graphviz_layout(srstr_graph, prog='neato')
else:
    st.write('Please select an option.')

draw_option = st.sidebar.radio(
    'Edges to display',
    ('Hierarchical', 'Hierarchical + Associative')
)

if draw_option == 'Hierarchical':
    to_draw = srstr_trees;
elif draw_option == 'Hierarchical + Associative':
    to_draw = srstr_graph;

data_or_graph = st.sidebar.radio(
    'Data or graph?',
    ('Graph', 'Data')
)

related_option = st.sidebar.multiselect(
    'Who is related to who',
    list(srstr_graph.nodes)
)

edge_x = []
edge_y = []
for node1, node2, attribute in to_draw.edges(data=True):
    if len(related_option) == 0 or node1 in related_option or node2 in related_option:
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
node_freq = []
if len(related_option) == 0:
    for node in to_draw.nodes:
        x, y = position[node]
        node_x.append(x)
        node_y.append(y)
        semantic_types.append(node)
        node_freq.append(sty_freq[node])
else:
    for node in related_option:
        x0, y0 = position[node]
        if x0 not in node_x and y0 not in node_y:
            node_x.append(x0)
            node_y.append(y0)
            semantic_types.append(node)
            node_freq.append(sty_freq[node])
        for index, row in srstr_sty_d[srstr_sty_d.STY1 == node].STY2.iteritems():
            x1, y1 = position[row]
            if x1 not in node_x and y1 not in node_y:
                node_x.append(x1)
                node_y.append(y1)
                semantic_types.append(row)
                node_freq.append(sty_freq[row])
        for index, row in srstr_sty_d[srstr_sty_d.STY2 == node].STY1.iteritems():
            x1, y1 = position[row]
            if x1 not in node_x and y1 not in node_y:
                node_x.append(x1)
                node_y.append(y1)
                semantic_types.append(row)
                node_freq.append(sty_freq[row])
        if data_or_graph == 'Data':
            st.subheader('Defined relations for: ' + str(node))
            st.write(srstr_sty_d[srstr_sty_d.STY1 == node])
            st.write(srstr_sty_d[srstr_sty_d.STY2 == node])

node_size_option = st.sidebar.radio(
    'Node size represents: (doesn\'t work yet)',
    ('Semantic type frequency in all defined relations', 'Concept amount in each semantic type')
)

color_option = st.sidebar.selectbox(
    'Choose your favorite node color',
    ('Hot', 'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu', 'Reds', 'Blues', 'Picnic',
     'Rainbow', 'Portland', 'Jet', 'Blackbody', 'Earth', 'Electric', 'Viridis')
)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    text=semantic_types,
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale=color_option,
        reversescale=True,
        color=node_freq,
        size=10,
        colorbar=dict(
            thickness=15,
            title=node_size_option,
            xanchor='left',
            titleside='right'
        ),
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
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20,l=5,r=5,t=40),
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
    )
)

fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
if data_or_graph == 'Graph':
    st.plotly_chart(fig, width=1000, height=700)
if data_or_graph == 'Data':
    st.subheader('All defined relations:')
    st.write(srstr_sty_d)
