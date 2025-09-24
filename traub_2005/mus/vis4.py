# vis.py ---
#
# Filename: vis.py
# Description:
# Author: Subhasis Ray
# Created: Sat Apr 26 09:22:59 2025 (+0530)
#

# Code:
"""Functions for visualizing a neuronal network"""
import sys
import igraph as ig
import pyvista as pv
import numpy as np
import time
from collections import defaultdict
import moose
from cortical_column import cell_counts
import adapter
import matplotlib.pyplot as plt


glyph_meshes = {
    'sphere': pv.Sphere(radius=10),
    'cone': pv.Cone(direction=(0, 0, 1), height=25, radius=15, resolution=20),
    'cylinder': pv.Cylinder(
        direction=(0, 0, 1), height=20, radius=10, resolution=20
    ),
    'octahedron': pv.Octahedron(radius=20),
}

#: Visualization spec. For each celltype a tuple:
#: (top, bottom, diameter, glyph, color)
cell_vis_spec = {
    'SupPyrRS': {
        'top': 100,
        'bottom': 900,
        # 'dia': 490,
        'dia': 700,
        'glyph': 'cone',
        'cmap': 'autumn',
    },
    'SupPyrFRB': {
        'top': 100,
        'bottom': 900,
        # 'dia': 490,
        'dia': 700,
        'glyph': 'cone',
        'cmap': 'spring',
    },
    'SupLTS': {
        'top': 100,
        'bottom': 900,
        # 'dia': 490,
        'dia': 700,
        'glyph': 'sphere',
        'cmap': 'hsv',
    },
    'SupAxoaxonic': {
        'top': 100,
        'bottom': 900,
        'dia': 490,
        'glyph': 'sphere',
        'cmap': 'Wistia',
    },
    'SupBasket': {
        'top': 100,
        'bottom': 900,
        # 'dia': 490,
        'dia': 700,
        'glyph': 'sphere',
        'cmap': 'winter',
    },
    'SpinyStellate': {
        'top': 1000,
        'bottom': 1600,
        # 'dia': 650,
        'dia': 900,
        'glyph': 'octahedron',
        'cmap': 'autumn',
    },
    'TuftedIB': {
        'top': 1700,
        'bottom': 2000,
        # 'dia': 550,
        'dia': 800,
        'glyph': 'cylinder',
        'cmap': 'Wistia',
    },
    'TuftedRS': {
        'top': 1700,
        'bottom': 2000,
        # 'dia': 550,
        'dia': 800,
        'glyph': 'cylinder',
        'cmap': 'spring',
    },
    'NontuftedRS': {
        'top': 2100,
        'bottom': 2500,
        # 'dia': 200,
        'dia': 500,
        'glyph': 'sphere',
        'cmap': 'autumn',
    },
    'DeepBasket': {
        'top': 2100,
        'bottom': 2500,
        'dia': 200,
        'glyph': 'sphere',
        'cmap': 'Wistia',
    },
    'DeepAxoaxonic': {
        'top': 2100,
        'bottom': 2500,
        # 'dia': 200,
        'dia': 500,
        'glyph': 'sphere',
        'cmap': 'summer',
    },
    'DeepLTS': {
        'top': 2100,
        'bottom': 2500,
        # 'dia': 200,
        'dia': 500,
        'glyph': 'sphere',
        'cmap': 'spring',
    },
    'TCR': {
        'top': 2800,
        'bottom': 3200,
        # 'dia': 200,
        'dia': 500,
        'glyph': 'cylinder',
        'cmap': 'autumn',
    },
    'nRT': {
        'top': 2800,
        'bottom': 3200,
        # 'dia': 200,
        'dia': 500,
        'glyph': 'sphere',
        'cmap': 'winter',
    },
}


def set_vis_attrs(graph, cell_counts=cell_counts, spec=cell_vis_spec):
    """Assign position to cells in graph"""
    graph.vs['color'] = [(0, 0, 0)] * len(graph.vs)
    for celltype, num in cell_counts.items():
        rpos = (
            np.random.uniform(low=0, high=1.0, size=num)
            * spec[celltype]['dia']
            / 2.0
        )
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)
        xpos = rpos * np.cos(theta)
        ypos = rpos * np.sin(theta)
        zpos = -np.random.uniform(
            low=spec[celltype]['top'], high=spec[celltype]['bottom'], size=num
        )
        vs = graph.vs.select(lambda v: v['name'].startswith(celltype))
        vs['pos'] = np.column_stack((xpos, ypos, zpos))
        vs['glyph'] = spec[celltype]['glyph']
        # vs['color'] = [spec[celltype]['color']] * len(vs)
        # print('#' * 10, vs['color'])

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


def display_data(
    datafile, celltype_attr=cell_vis_spec, vmin=-70e-3, vmax=10e-3
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
    shape = (1, 3)  # 14x3 grid
    # shape = (len(celltype_attr), 3)  # 14x3 grid
    col_weights = [0.5, 1, 0.5]
    groups = [
        (np.s_[:], 1)
    ]  # Middle column spans all rows and displays the 3D view
    plotter = pv.Plotter(
        shape=shape,
        col_weights=col_weights,
        groups=groups,
        border=False,
    )
    plotter.subplot(0, 1)
    glyph_actors = {}
    pdata_dict = {}
    glyph_dict = {}
    for celltype, vinfo in celltype_attr.items():
        mesh = glyph_meshes[vinfo['glyph']]
        vs = graph.vs.select(lambda v: v['celltype'] == celltype)
        pdata = pv.PolyData(vs['pos'])
        pdata['Vm'] = [0.0] * pdata.n_points
        pdata.set_active_scalars('Vm')
        pdata_dict[celltype] = pdata
        glyphs = pdata.glyph(scale=False, factor=2, geom=mesh)
        glyphs.set_active_scalars('Vm')
        glyph_dict[celltype] = glyphs
        actor = plotter.add_mesh(
            glyphs, cmap=vinfo['cmap'], show_scalar_bar=False
        )
        actor.mapper.scalar_visibility = True
        actor.mapper.scalar_range = (-100e-3, 0)
        glyph_actors[celltype] = actor
    plotter.set_background('black')
    plotter.camera_position = [
        (0.0, 10.0, -1200.0),  # position
        (0.0, 0.0, -1200.0),  # focal point
        (0.0, 1.0, 0.0),
    ]
    plotter.reset_camera()
    plotter.enable_depth_peeling()
    celltype_data = defaultdict(dict)
    for cell_name, vm in datadict.items():
        celltype_data[cell_name.partition('_')[0]][
            int(cell_name.rpartition('_')[-1])
        ] = [vm]
    charts = []
    for plot_no in (0, 2):
        plotter.subplot(0, plot_no)
        chart = pv.Chart2D()
        plotter.add_chart(chart)
        chart.y_range = (-1.0, len(celltype_attr))
        chart.x_range = (0, 1.0)
        chart.grid = False
        charts.append(chart)
        for ii, (celltype, cell_data) in enumerate(celltype_data.items()):
            x = np.arange(1)
            y = np.arange(1)
            _ = chart.line(x, y, color='red', width=1)
        print('Added chart for subplot', plot_no)

    def update(step):
        t, newdata = next(data)
        # print('Step', step, 'Time', t)
        vm_dict = defaultdict(list)
        for cell_name, vm in newdata.items():
            celltype = cell_name.partition('_')[0]
            vm_dict[celltype].append(vm)
            index = int(cell_name.rpartition('_')[-1])
            vmlist = celltype_data[celltype][index]
            vmlist.append(vm)
        for chart in charts:
            chart.clear()

        vm_plot_scale = 7
        for ii, (celltype, vmlist) in enumerate(vm_dict.items()):
            glyph = glyph_meshes[celltype_attr[celltype]['glyph']]
            glyph_dict[celltype]['Vm'] = np.array(vmlist).repeat(glyph.n_cells)
            actor = glyph_actors[celltype]
            actor.rotate_z(0.5)
            vm0 = np.array(celltype_data[celltype][0]) * vm_plot_scale
            cmap = celltype_attr[celltype]['cmap']
            # print(celltype, 'Colormap', cmap)
            color = plt.get_cmap(cmap)(0.0)
            for chart in charts:
                _ = chart.line(
                    np.arange(len(vm0)) * 1e-3, vm0 + ii, color=color
                )
                print('Added plot for', ii, celltype, vm0[:5])

    plotter.iren.initialize()
    plotter.add_timer_event(max_steps=1000, duration=1, callback=update)
    print('Here ....')
    plotter.show()


def display_spike_data(
    datafile,
    celltype_attr=cell_vis_spec,
    vmin=-100e-3,
    vmax=0,
    cell_mult=1,
    moviefile=None,
):
    """cell_mult: make these many copy of each cell to make visual appealing. This is unscientific and merely for creating media."""
    datadict = adapter.get_spike_vm(datafile, amp=1000e-3, baseline=-65e-3)
    graph = ig.Graph(directed=True)
    cell_counts = defaultdict(int)
    cell_clones = {}
    for cell_name in datadict:
        celltype = cell_name.partition('_')[0]
        graph.add_vertex(name=cell_name, celltype=celltype)
        # Make fake cells by cloning
        for ii in range(cell_mult - 1):
            graph.add_vertex(name=f'{cell_name}_{ii}', celltype=celltype)

        cell_counts[celltype] += 1 + (cell_mult - 1)
    set_vis_attrs(graph, cell_counts=cell_counts, spec=celltype_attr)
    shape = (1, 3)  # 14x3 grid
    # shape = (len(celltype_attr), 3)  # 14x3 grid
    col_weights = [0.5, 1, 0.5]
    groups = [
        (np.s_[:], 1)
    ]  # Middle column spans all rows and displays the 3D view
    plotter = pv.Plotter(
        shape=shape,
        col_weights=col_weights,
        groups=groups,
        # border=True,
        # border_color='white',
        border=False,
        window_size=(1280, 720),
    )
    if moviefile is not None:
        plotter.open_movie(moviefile, framerate=30)
    plotter.subplot(0, 1)
    glyph_actors = {}
    pdata_dict = {}
    glyph_dict = {}
    for celltype, vinfo in celltype_attr.items():
        mesh = glyph_meshes[vinfo['glyph']]
        vs = graph.vs.select(lambda v: v['celltype'] == celltype)
        pdata = pv.PolyData(vs['pos'])
        pdata['Vm'] = [0.0] * pdata.n_points
        pdata.set_active_scalars('Vm')
        pdata_dict[celltype] = pdata
        glyphs = pdata.glyph(scale=False, factor=2, geom=mesh)
        glyphs['Vm'] = [0] * glyphs.n_cells
        glyphs.set_active_scalars('Vm')
        glyph_dict[celltype] = glyphs
        actor = plotter.add_mesh(
            glyphs,
            cmap=vinfo['cmap'],
            clim=(0, 1),
            opacity=0.7,
            show_scalar_bar=False,
            # flip_scalars=True,
        )
        # actor = plotter.add_mesh(glyphs, scalars='Vm', opacity='linear', use_transparency=True, cmap=vinfo['cmap'])
        actor.mapper.scalar_visibility = True
        # actor.mapper.scalar_range = (-100e-3, 0)
        glyph_actors[celltype] = actor
    plotter.set_background('black')
    plotter.camera.position = (0.0, 400.0, -1600.0)
    plotter.camera.focal_point = (0.0, -400.0, -1600.0)
    plotter.camera.view_up = (0.0, 1.0, 0.0)
    plotter.reset_camera()
    # plotter.camera.tight()
    plotter.enable_depth_peeling()
    celltype_data = defaultdict(dict)
    for cell_name, vm in datadict.items():
        celltype = cell_name.partition('_')[0]
        celltype_data[celltype][cell_name] = (vm - vmin) / (vmax - vmin)
    charts = []
    for plot_no in (0, 2):
        plotter.subplot(0, plot_no)
        chart = pv.Chart2D()
        plotter.add_chart(chart)
        chart.y_range = (-1.0, len(celltype_attr))
        chart.x_range = (0, 1.0)
        chart.grid = False
        charts.append(chart)
        for ii, (celltype, cell_data) in enumerate(celltype_data.items()):
            x = np.arange(1)
            y = np.arange(1)
            _ = chart.line(x, y, color='red', width=1)
        print('Added chart for subplot', plot_no)
    if moviefile:
        plotter.write_frame()

    def update(step):
        for chart in charts:
            chart.clear()
        vm_plot_scale = 1
        for ii, (celltype, vmdict) in enumerate(celltype_data.items()):
            glyph = glyph_meshes[celltype_attr[celltype]['glyph']]
            vm_instant = [
                vm[step] for vm in vmdict.values() for jj in range(cell_mult)
            ]
            glyph_dict[celltype]['Vm'] = np.array(vm_instant).repeat(
                glyph.n_cells
            )
            cmap = plt.get_cmap(celltype_attr[celltype]['cmap'])
            # print(celltype, vm_instant[0], cmap(vm_instant[0]))
            actor = glyph_actors[celltype]
            actor.rotate_z(0.5)
            key0 = list(vmdict.keys())[0]
            vm0 = np.array(vmdict[key0][:step]) * vm_plot_scale
            cmap = celltype_attr[celltype]['cmap']
            # print(celltype, 'Colormap', cmap)
            color = plt.get_cmap(cmap)(0)
            for chart in charts:
                _ = chart.line(
                    np.arange(len(vm0)) * 1e-3, vm0 + ii, color=color
                )
        if moviefile and (step % 10 == 0):
            plotter.write_frame()

    plotter.iren.initialize()
    plotter.add_timer_event(max_steps=1000, duration=1, callback=update)
    print('Here ....')
    # plotter.show()
    plotter.show(auto_close=False)
    plotter.close()


if __name__ == '__main__':
    fpath = '../../../traub_2005_full/dataviz/test_data/data_20111025_115951_4849.h5'
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
    if len(sys.argv) > 2:
        cell_mult = int(sys.argv[2])
    else:
        cell_mult = 1
    # display_data(fpath)
    display_spike_data(fpath, cell_mult=cell_mult)
#
# vis.py ends here
