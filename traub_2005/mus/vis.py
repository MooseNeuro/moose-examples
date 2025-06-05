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
from collections import defaultdict
import moose
from cortical_column import cell_counts
import adapter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize


norm = Normalize(vmin=-100e-3, vmax=0.0)

glyph_meshes = {
    'sphere': pv.Sphere(radius=10),
    'cone': pv.Cone(direction=(0, 0, 1), height=30, radius=20, resolution=20),
    'cylinder': pv.Cylinder(
        direction=(0, 0, 1), height=20, radius=10, resolution=20
    ),
}

#: Visualization spec. For each celltype a tuple:
#: (top, bottom, diameter, glyph, color)
cell_vis_spec = {
    'SupPyrRS': {
        'top': 100,
        'bottom': 900,
        'dia': 490,
        'glyph': 'cone',
        'color': '#924900',
    },
    'SupPyrFRB': {
        'top': 100,
        'bottom': 900,
        'dia': 490,
        'glyph': 'cone',
        'color': '#920000',
    },
    'SupLTS': {
        'top': 100,
        'bottom': 900,
        'dia': 490,
        'glyph': 'sphere',
        'color': '#b6dbff',
    },
    'SupAxoaxonic': {
        'top': 100,
        'bottom': 900,
        'dia': 490,
        'glyph': 'sphere',
        'color': '#b66dff',
    },
    'SupBasket': {
        'top': 100,
        'bottom': 900,
        'dia': 490,
        'glyph': 'sphere',
        'color': '#6db6ff',
    },
    'SpinyStellate': {
        'top': 1000,
        'bottom': 1600,
        'dia': 650,
        'glyph': 'sphere',
        'color': '#006ddb',
    },
    'TuftedIB': {
        'top': 1700,
        'bottom': 2000,
        'dia': 550,
        'glyph': 'cylinder',
        'color': '#24ff24',
    },
    'TuftedRS': {
        'top': 1700,
        'bottom': 2000,
        'dia': 550,
        'glyph': 'cylinder',
        'color': '#ffff6d',
    },
    'NontuftedRS': {
        'top': 2100,
        'bottom': 2500,
        'dia': 200,
        'glyph': 'sphere',
        'color': '#ffb6db',
    },
    'DeepBasket': {
        'top': 2100,
        'bottom': 2500,
        'dia': 200,
        'glyph': 'sphere',
        'color': '#009292',
    },
    'DeepAxoaxonic': {
        'top': 2100,
        'bottom': 2500,
        'dia': 200,
        'glyph': 'sphere',
        'color': '#004949',
    },
    'DeepLTS': {
        'top': 2100,
        'bottom': 2500,
        'dia': 200,
        'glyph': 'sphere',
        'color': '#ff6db6',
    },
    'TCR': {
        'top': 2800,
        'bottom': 3200,
        'dia': 200,
        'glyph': 'sphere',
        'color': '#db6d00',
    },
    'nRT': {
        'top': 2800,
        'bottom': 3200,
        'dia': 200,
        'glyph': 'sphere',
        'color': '#490092',
    },
}


# def get_lut(spec, n_values=256, vmin=-100e-3, vmax=0.0):
#     cstring = spec['color']
#     color = [(
#             int(cstring[1:3], 16),
#             int(cstring[3:5], 16),
#             int(cstring[5:7], 16),
#     )] * n_values
#     alpha = np.arange(n_values) * 256.0 / n_values
#     color = np.column_stack((color, alpha.astype(int)))
#     lut = pv.LookupTable(values=color,  scalar_range=(vmin, vmax), alpha_range=(0.1, 1))
#     return lut

def get_lut(spec, n_values=256, vmin=-100e-3, vmax=0.0):
    cstring = spec['color']    
    red, green, blue = (
            int(cstring[1:3], 16),
            int(cstring[3:5], 16),
            int(cstring[5:7], 16),
    )
    print(red, green, blue)
    red = np.linspace(0, red, n_values)
    green = np.linspace(0, green, n_values)
    blue = np.linspace(0, blue, n_values)
    alpha = np.arange(n_values) * 256.0 / n_values
    color = np.column_stack((red, green, blue, alpha))
    print('%'* 10, color[-1])
    lut = pv.LookupTable(values=color.astype(int),  scalar_range=(vmin, vmax), alpha_range=(0, 1))
    return lut


def set_vis_attrs(graph, cell_counts=cell_counts, spec=cell_vis_spec):
    """Assign position to cells in graph"""
    graph.vs['color'] = [(0, 0, 0)] * len(graph.vs)
    for celltype, num in cell_counts.items():
        rpos = np.random.uniform(low=0, high=1.0, size=num) * spec[celltype]['dia'] / 2.0
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)
        xpos = rpos * np.cos(theta)
        ypos = rpos * np.sin(theta)
        zpos = -np.random.uniform(low=spec[celltype]['top'], high=spec[celltype]['bottom'], size=num)
        vs = graph.vs.select(lambda v: v['name'].startswith(celltype))
        vs['pos'] = np.column_stack((xpos, ypos, zpos))
        vs['glyph'] = spec[celltype]['glyph']
        vs['color'] = [spec[celltype]['color']] * len(vs)
        print('#' * 10, vs['color'])

    return graph


def display_network(
    graph, cell_counts=cell_counts, celltype_attr=cell_vis_spec
):
    """Display the network in 3D where node in `graph` represent cells
    and edges, synapses."""
    tstart = time.perf_counter()
    set_vis_attrs(graph, cell_counts=cell_counts, spec=celltype_attr)
    # print(graph.vs['color'])
    edge_mesh = pv.PolyData(graph.vs['pos'])
    edge_mesh['color'] = graph.vs['color']
    edge_list = graph.get_edgelist()
    edge_mesh.lines = np.array([(2, edge[0], edge[1]) for edge in edge_list])
    plotter = pv.Plotter()
    # plotter.add_mesh(edge_mesh, scalars=np.array(graph.vs['color']), rgb=True, opacity=0.1)
    glyph_actors = {}
    for celltype, vinfo in celltype_attr.items():
        glyph_name, color = vinfo[3], vinfo[4]
        mesh = glyph_meshes[glyph_name]
        vs = graph.vs.select(lambda v: v['celltype'] == celltype)
        glyphs = pv.PolyData(vs['pos']).glyph(
            scale=False, factor=10, geom=mesh
        )
        actor = plotter.add_mesh(glyphs, color=color, opacity=1.0)
        glyph_actors[celltype] = actor
    plotter.set_background('black')
    # plotter.add_axes()
    plotter.camera_position = [
        (0.0, 500.0, -1200.0),  # position
        (0.0, 0.0, -1200.0),  # focal point
    ]
    plotter.reset_camera()

    tend = time.perf_counter()
    print(f'Created  visualization in {tend - tstart} s')
    plotter.show()


def display_data(datafile, celltype_attr=cell_vis_spec, vmin=-100e-3, vmax=0):
    data = adapter.get_data(datafile, field='Vm')
    t, datadict = next(data)
    graph = ig.Graph(directed=True)
    cell_counts = defaultdict(int)
    for cell_name in datadict:
        celltype = cell_name.partition('_')[0]
        graph.add_vertex(name=cell_name, celltype=celltype)
        cell_counts[celltype] += 1
    set_vis_attrs(graph, cell_counts=cell_counts, spec=celltype_attr)
    plotter = pv.Plotter()
    glyph_actors = {}
    pdata_dict = {}
    for celltype, vinfo in celltype_attr.items():
        mesh = glyph_meshes[vinfo['glyph']]
        vs = graph.vs.select(lambda v: v['celltype'] == celltype)
        for vertex in vs:
            actor = plotter.add_mesh(
                mesh.copy().translate(vertex['pos']), color=vinfo['color']
            )
            glyph_actors[vertex['name']] = actor
    plotter.set_background('black')
    plotter.camera_position = [
        (0.0, 500.0, -1200.0),  # position
        (0.0, 0.0, -1200.0),  # focal point
        (0.0, 1.0, 0.0),  # view-up vector, Y is up
    ]
    plotter.reset_camera()
    plotter.enable_depth_peeling()

    def update(step):
        t, newdata = next(data)
        print('Step', step, 'Time', t)
        for cell_name, vm in newdata.items():
            celltype = cell_name.partition('_')[0]
            v = min(255, int(255 * (vm - vmin) / (vmax - vmin)))
            color = f'{celltype_attr[celltype]["color"]}{v:02x}'
            actor = glyph_actors[cell_name]
            actor.prop.color = color
        plotter.render()

    plotter.iren.initialize()
    plotter.add_timer_event(max_steps=1000, duration=1, callback=update)
    print('Here ....')
    plotter.show()


def display_data_2(
    datafile, celltype_attr=cell_vis_spec, vmin=-100e-3, vmax=0
):
    data = adapter.get_data(datafile, field='Vm')
    t, datadict = next(data)
    graph = ig.Graph(directed=True)
    cell_counts = defaultdict(int)
    for cell_name in datadict:
        celltype = cell_name.partition('_')[0]
        graph.add_vertex(name=cell_name, celltype=celltype)
        cell_counts[celltype] += 1
    set_vis_attrs(graph, cell_counts=cell_counts, spec=celltype_attr)
    plotter = pv.Plotter()
    glyph_actors = {}
    pdata_dict = {}
    glyph_dict = {}
    lut_dict = {}
    for celltype, vinfo in celltype_attr.items():
        mesh = glyph_meshes[vinfo['glyph']]
        vs = graph.vs.select(lambda v: v['celltype'] == celltype)
        pdata = pv.PolyData(vs['pos'])
        # pdata.point_data['Vm'] = [0.0] * pdata.n_points
        # pdata.set_active_scalars('Vm')
        pdata_dict[celltype] = pdata
        glyphs = pdata.glyph(scale=False, factor=1, geom=mesh)
        glyph_dict[celltype] = glyphs
        lut = get_lut(vinfo)
        lut_dict[celltype] = lut
        actor = plotter.add_mesh(glyphs)
        actor.mapper.lookup_table = None
        actor.mapper.scalar_visibility = True
        glyph_actors[celltype] = actor
    plotter.set_background('black')
    # plotter.add_axes()
    plotter.camera_position = [
        (0.0, 500.0, -1200.0),  # position
        (0.0, 0.0, -1200.0),  # focal point
        (0.0, 1.0, 0.0),
    ]
    plotter.reset_camera()
    plotter.enable_depth_peeling()

    def update(step):
        t, newdata = next(data)
        print('Step', step, 'Time', t)
        vm_dict = defaultdict(list)
        for cell_name, vm in newdata.items():
            celltype = cell_name.partition('_')[0]
            vm_dict[celltype].append(vm)
        for celltype, vmlist in vm_dict.items():
            # if not 'Pyr' in celltype:
            #     continue
            # pdata_dict[celltype].point_data['colors'] = colors[celltype]
            actor = glyph_actors[celltype]
            actor.rotate_z(0.5)
            lut = lut_dict[celltype]
            ds = actor.mapper.dataset
            orig_glyph = glyph_meshes[celltype_attr[celltype]['glyph']]
            colors = np.vstack([lut(vm) for vm in vmlist])
            ds.cell_data['colors'] = colors.repeat(orig_glyph.n_cells, axis=0)

            # actor.mapper.dataset.cell_data['colors'] = colors[celltype]
            # print(pdata_dict[celltype]['Vm'])

    plotter.iren.initialize()
    plotter.add_timer_event(max_steps=1000, duration=1, callback=update)
    print('Here ....')
    plotter.show()


if __name__ == '__main__':
    fpath = '../../../traub_2005_full/dataviz/test_data/data_20111025_115951_4849.h5'
    display_data_2(fpath)

#
# vis.py ends here
