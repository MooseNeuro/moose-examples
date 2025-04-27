# vis.py ---
#
# Filename: vis.py
# Description:
# Author: Subhasis Ray
# Created: Sat Apr 26 09:22:59 2025 (+0530)
#

# Code:
"""Functions for visualizing a neuronal network"""
import igraph as ig
import pyvista as pv
import numpy as np
import time
import moose
from cortical_column import cell_counts


glyph_meshes = {
    'sphere': pv.Sphere(radius=1),
    'cone': pv.Cone(
        direction=(0, 0, 1), height=2, radius=1, resolution=20
    ),
    'cylinder': pv.Cylinder(
        direction=(0, 0, 1), height=2, radius=1, resolution=20
    ),
}

#: Visualization spec. For each celltype a tuple:
#: (top, bottom, diameter, glyph, color)
cell_vis_spec = {
    'SupPyrRS': (100, 900, 490, 'cone', '#924900'),
    'SupPyrFRB': (100, 900, 490, 'cone', '#920000'),
    'SupLTS': (100, 900, 490, 'sphere', '#b6dbff'),
    'SupAxoaxonic': (100, 900, 490, 'sphere', '#b66dff'),
    'SupBasket': (100, 900, 490, 'sphere', '#6db6ff'),
    'SpinyStellate': (1000, 1600, 650, 'sphere', '#006ddb'),
    'TuftedIB': (1700, 2000, 550, 'cylinder', '#24ff24'),
    'TuftedRS': (1700, 2000, 550, 'cylinder', '#ffff6d'),
    'NontuftedRS': (2100, 2500, 200, 'sphere', '#ffb6db'),
    'DeepBasket': (2100, 2500, 200, 'sphere', '#009292'),
    'DeepAxoaxonic': (2100, 2500, 200, 'sphere', '#004949'),
    'DeepLTS': (2100, 2500, 200, 'sphere', '#ff6db6'),
    'TCR': (2800, 3200, 200, 'sphere', '#db6d00'),
    'nRT': (2800, 3200, 200, 'sphere', '#490092'),
}


def make_graph(model_root):
    tstart = time.perf_counter()
    model_root = moose.element(model_root)
    model_graph = ig.Graph(directed=True)
    for element in model_root.children:
        celltype = element.name.partition('_')[0]
        model_graph.add_vertex(
            name=element.name,
            celltype=celltype,
        )
    for spike in moose.wildcardFind(f'{model_root.path}/##/spike'):
        pre = spike.parent.parent.name
        print('#' * 10, pre)
        for syn in moose.neighbors(spike, 'spikeOut'):
            post = syn.parent.parent.parent.parent.name
            model_graph.add_edge(pre, post)
            print(pre, post)
    tend = time.perf_counter()
    print(f'Created graph in {tend - tstart} s')
    return model_graph


def set_vis_attrs(graph, cell_counts=cell_counts, spec=cell_vis_spec):
    """Assign position to cells in graph"""
    graph.vs['color'] = [(0, 0, 0)] * len(graph.vs)
    for celltype, num in cell_counts.items():
        top, bottom, dia, glyph, color = spec[celltype]
        print('$' * 10, top, bottom, 'dia', dia, glyph, color)
        rpos = np.random.uniform(low=0, high=1.0, size=num) * dia / 2.0
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)
        xpos = rpos * np.cos(theta)
        ypos = rpos * np.sin(theta)
        zpos = -np.random.uniform(low=top, high=bottom, size=num)
        vs = graph.vs.select(lambda v: v['name'].startswith(celltype))
        vs['pos'] = np.column_stack((xpos, ypos, zpos))
        vs['glyph'] = glyph
        vs['color'] = [(
            int(color[1:3], 16) ,
            int(color[3:5], 16) ,
            int(color[5:7], 16) ,
        )] * len(vs)
        print('#'* 10, vs['color'])

    return graph


def display_network(
    graph, cell_counts=cell_counts, celltype_attr=cell_vis_spec
):
    """Display the network in 3D where node in `graph` represent cells
    and edges, synapses."""
    tstart = time.perf_counter()
    set_vis_attrs(graph, cell_counts=cell_counts, spec=cell_vis_spec)
    # print(graph.vs['color'])
    edge_mesh = pv.PolyData(graph.vs['pos'])
    edge_mesh['color'] = graph.vs['color']
    edge_list = graph.get_edgelist()
    edge_mesh.lines = np.array([(2, edge[0], edge[1]) for edge in edge_list])
    plotter = pv.Plotter()
    # plotter.add_mesh(edge_mesh, scalars=np.array(graph.vs['color']), rgb=True, opacity=0.1)
    glyph_actors = {}
    for celltype, vinfo in cell_vis_spec.items():
        glyph_name, color = vinfo[3], vinfo[4]
        mesh = glyph_meshes[glyph_name]
        vs = graph.vs.select(lambda v: v['celltype'] == celltype)
        glyphs = pv.PolyData(vs['pos']).glyph(scale=False, factor=10, geom=mesh)
        actor = plotter.add_mesh(glyphs, color=color, opacity=1.0)
        glyph_actors[celltype] = actor
    plotter.set_background('black')
    # plotter.add_axes()
    tend = time.perf_counter()
    print(f'Created  visualization in {tend - tstart} s')
    plotter.show()


#
# vis.py ends here
