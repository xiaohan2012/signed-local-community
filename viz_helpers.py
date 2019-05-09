
import networkx as nx
from helpers import (
    get_lcc, pos_spring_layout, signed_layout, get_borderless_fig,
    draw_edges
)


def draw_query_result(g, C1, C2, layout='pos', show_query=False, r=None, use_lcc=False):
    assert layout in {'pos', 'spectral'}
    subg = g.subgraph(list(C1) + list(C2))
    if use_lcc:
        print('use largest CC')
        subg = get_lcc(subg)
        new_nodes = set(subg.nodes())
        C1 = set(C1).intersection(new_nodes)
        C2 = set(C2).intersection(new_nodes)

    mapping = {n: i for i, n in enumerate(subg.nodes())}
    subg = nx.relabel_nodes(subg, mapping=mapping)

    C1 = [mapping[n] for n in C1]
    C2 = [mapping[n] for n in C2]

    if layout == 'pos':
        pos = pos_spring_layout(subg)
    else:
        pos = signed_layout(subg)

    fig, ax = get_borderless_fig()
    # draw_nodes(subg, pos, ax=ax)
    styles = dict(
        node_size=40,
        linewidths=0,
        alpha=0.9,
    )

    nx.draw_networkx_nodes(
        subg, pos,
        nodelist=C1,
        node_shape='v',
        node_color='orange',
        **styles
    )
    nx.draw_networkx_nodes(
        subg, pos,
        nodelist=C2,
        node_shape='8',
        node_color='black',
        **styles
    )
    if show_query:
        nx.draw_networkx_nodes(subg, pos, nodelist=[mapping[r['query']]],
                               node_size=80, linewidths=0,
                               node_color='green',
                               node_shape='s')
    draw_edges(subg, pos, ax=ax, width=1.0, alpha=0.5)
    return fig, ax
