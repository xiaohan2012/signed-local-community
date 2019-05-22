
import networkx as nx
from helpers import (
    get_lcc, pos_spring_layout, signed_layout, get_borderless_fig,
    draw_edges
)


def draw_query_result(
        g, C1, C2,
        layout='pos',
        seeds1=None, seeds2=None,
        C1_labels=None, C2_labels=None,
        use_lcc=False,
        label_x_offset=0,
        label_y_offset=0,
        label_font_size=16
):
    assert layout in {'pos', 'spectral', 'spring'}
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
    elif layout == 'spring':
        pos = nx.fruchterman_reingold_layout(subg, weight='sign')
    else:
        pos = signed_layout(subg)

    fig, ax = get_borderless_fig()
    # draw_nodes(subg, pos, ax=ax)
    styles = dict(
        node_size=100,
        linewidths=0,
        alpha=0.5,
    )

    nx.draw_networkx_nodes(
        subg, pos,
        nodelist=C1,
        node_shape='8',
        node_color='cyan',
        **styles
    )
    nx.draw_networkx_nodes(
        subg, pos,
        nodelist=C2,
        node_shape='8',
        node_color='magenta',
        **styles
    )

    label_pos = {k: (v[0]+label_x_offset, v[1]+label_y_offset) for k, v in pos.items()}
    if C1_labels:
        C1_labels = {mapping[i]: l for i, l in C1_labels.items()}
        nx.draw_networkx_labels(
            subg, label_pos, nodelist=C1_labels.keys(), labels=C1_labels,
            font_size=label_font_size,
        )

    if C2_labels:
        C2_labels = {mapping[i]: l for i, l in C2_labels.items()}
        nx.draw_networkx_labels(
            subg, label_pos, nodelist=C2_labels.keys(), labels=C2_labels,
            font_size=label_font_size,
        )

    if seeds1:
        nx.draw_networkx_nodes(subg, pos, nodelist=[mapping[s] for s in seeds1],
                               node_size=120, linewidths=0,
                               node_color='green',
                               node_shape='s',
        )
    if seeds2:
        nx.draw_networkx_nodes(subg, pos, nodelist=[mapping[s] for s in seeds2],
                               node_size=120, linewidths=0,
                               node_color='blue',
                               node_shape='s',
        )
    draw_edges(subg, pos, ax=ax, width=1.0, alpha=0.5)
    return fig, ax
