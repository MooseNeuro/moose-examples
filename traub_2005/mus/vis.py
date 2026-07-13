# vis.py ---
#
# Filename: vis.py
# Description:
# Author: Subhasis Ray
# Created: Sat Apr 26 09:22:59 2025 (+0530)
#

# Code:
"""Functions for visualizing a neuronal network"""
import colorsys
import igraph as ig
import pyvista as pv
import numpy as np
import time
from collections import defaultdict
import moose
from cortical_column import cell_counts
import adapter
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    to_rgb,
)
from matplotlib import colormaps


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
    """Assign a 3D position, glyph and color to every cell in `graph`.

    Cells are laid out in a cylindrical shell per cell type using the
    depth range and diameter from `spec`. The number of cells of each
    type is taken from the graph itself so this works regardless of the
    scale the network was built at.
    """
    graph.vs['color'] = [(0, 0, 0)] * len(graph.vs)
    for celltype in spec:
        vs = graph.vs.select(celltype_eq=celltype)
        num = len(vs)
        if num == 0:
            continue
        rpos = np.random.uniform(low=0, high=1.0, size=num) * spec[celltype]['dia'] / 2.0
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)
        xpos = rpos * np.cos(theta)
        ypos = rpos * np.sin(theta)
        zpos = -np.random.uniform(low=spec[celltype]['top'], high=spec[celltype]['bottom'], size=num)
        vs['pos'] = np.column_stack((xpos, ypos, zpos))
        vs['glyph'] = spec[celltype]['glyph']
        vs['color'] = [spec[celltype]['color']] * num
    return graph


def display_network(
    graph, cell_counts=cell_counts, celltype_attr=cell_vis_spec
):
    """Display the network in 3D where node in `graph` represent cells
    and edges, synapses."""
    tstart = time.perf_counter()
    set_vis_attrs(graph, cell_counts=cell_counts, spec=celltype_attr)
    positions = np.array(graph.vs['pos'])
    edge_mesh = pv.PolyData(positions)
    edge_mesh['color'] = graph.vs['color']
    edge_list = graph.get_edgelist()
    if edge_list:
        edge_mesh.lines = np.array(
            [(2, edge[0], edge[1]) for edge in edge_list]
        )
    plotter = pv.Plotter()
    # plotter.add_mesh(edge_mesh, scalars=np.array(graph.vs['color']), rgb=True, opacity=0.1)
    glyph_actors = {}
    for celltype, vinfo in celltype_attr.items():
        glyph_name, color = vinfo['glyph'], vinfo['color']
        mesh = glyph_meshes[glyph_name]
        vs = graph.vs.select(celltype_eq=celltype)
        if len(vs) == 0:
            continue
        glyphs = pv.PolyData(np.array(vs['pos'])).glyph(
            scale=False, factor=10, geom=mesh
        )
        actor = plotter.add_mesh(glyphs, color=color, opacity=1.0)
        glyph_actors[celltype] = actor
    plotter.set_background('black')
    # plotter.add_axes()
    plotter.camera_position = [
        (0.0, 500.0, -1200.0),  # position
        (0.0, 0.0, -1200.0),  # focal point
        (0.0, 1.0, 0.0),  # view-up vector, Y is up
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
            v = max(0, min(255, int(255 * (vm - vmin) / (vmax - vmin))))
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


def display_activity(
    datafile,
    celltype_attr=cell_vis_spec,
    field='Vm',
    vmin=-100e-3,
    vmax=40e-3,
    window=50e-3,
    gamma=0.6,
    interval=20,
):
    """Animate network activity from a recorded NSDF file.

    Each cell type keeps its own hue (the same color used for its trace
    in the right panel); the membrane potential (`field`) modulates the
    brightness of that hue, so a glyph brightens as its cell
    depolarizes. The brightness is `norm(Vm) ** gamma` (a smaller
    `gamma` lifts sub-threshold activity into view) mapped over the
    range [`vmin`, `vmax`]. The right panel shows one scrolling Vm trace
    per cell type, stacked top-to-bottom by cortical depth and advancing
    right-to-left over the most recent `window` seconds.

    `interval` is the timer period in milliseconds between animation
    frames.
    """
    data = adapter.get_data(datafile, field=field)
    t0, frame0 = next(data)
    t1, frame1 = next(data)
    dt = (t1 - t0) if t1 > t0 else 1.0
    buflen = max(2, int(round(window / dt)))

    # Lay the recorded cells out by type
    graph = ig.Graph(directed=True)
    counts = defaultdict(int)
    for cell_name in frame0:
        celltype = cell_name.partition('_')[0]
        graph.add_vertex(name=cell_name, celltype=celltype)
        counts[celltype] += 1
    set_vis_attrs(graph, cell_counts=counts, spec=celltype_attr)

    norm = Normalize(vmin=vmin, vmax=vmax)
    # Hue and saturation of each cell type's base color; Vm sets the
    # value (brightness). A floor keeps resting cells faintly visible.
    bright_floor = 0.12
    base_hs = {
        ct: colorsys.rgb_to_hsv(*to_rgb(spec['color']))[:2]
        for ct, spec in celltype_attr.items()
    }

    def vm_to_rgb(celltype, vm):
        nvm = float(min(1.0, max(0.0, norm(vm))))
        bright = bright_floor + (1.0 - bright_floor) * (nvm ** gamma)
        hue, sat = base_hs[celltype]
        return colorsys.hsv_to_rgb(hue, sat, bright)

    plotter = pv.Plotter(shape=(1, 2))

    # ---- Left panel: 3D glyphs, hue per type and brightness by Vm ----
    plotter.subplot(0, 0)
    glyph_actors = {}
    for celltype in celltype_attr:
        vs = graph.vs.select(celltype_eq=celltype)
        if len(vs) == 0:
            continue
        mesh = glyph_meshes[celltype_attr[celltype]['glyph']]
        for vertex in vs:
            actor = plotter.add_mesh(
                mesh.copy().translate(vertex['pos']),
                color=vm_to_rgb(celltype, frame0[vertex['name']]),
            )
            glyph_actors[vertex['name']] = actor
    # A zero-opacity grayscale reference to draw a brightness (Vm) bar
    ref = pv.PolyData(np.zeros((2, 3)))
    ref[field] = np.array([vmin, vmax])
    plotter.add_mesh(
        ref,
        scalars=field,
        cmap='gray',
        clim=[vmin, vmax],
        opacity=0.0,
        show_scalar_bar=True,
        scalar_bar_args={'title': f'{field} (V)', 'color': 'white'},
    )
    plotter.set_background('black')
    plotter.camera_position = [
        (0.0, 500.0, -1200.0),
        (0.0, 0.0, -1200.0),
        (0.0, 1.0, 0.0),
    ]
    plotter.reset_camera()

    # ---- Right panel: one scrolling Vm axis per cell type, stacked
    # top-to-bottom in order of cortical depth ----
    plotter.subplot(0, 1)
    # one representative cell per type
    reps = {}
    for cell_name in frame0:
        celltype = cell_name.partition('_')[0]
        if celltype not in reps:
            reps[celltype] = cell_name
    # order most-superficial (smallest depth) first, so it sits on top
    ordered = sorted(reps.items(), key=lambda kv: celltype_attr[kv[0]]['top'])
    n = len(ordered)
    slot = 0.98 / n
    times = []
    traces = {}
    lines = {}
    charts = []
    for i, (celltype, cell_name) in enumerate(ordered):
        chart = pv.Chart2D()
        chart.size = (0.94, slot * 0.86)
        # i == 0 is the most superficial layer -> highest on screen
        chart.loc = (0.02, 0.01 + (n - 1 - i) * slot)
        chart.y_range = [vmin, vmax]
        chart.y_label = celltype
        chart.y_axis.tick_labels_visible = False
        if i < n - 1:
            # only the bottom-most axis carries the shared time scale
            chart.x_axis.tick_labels_visible = False
            chart.x_axis.label = ''
        else:
            chart.x_axis.label = 'Time (s)'
        # Make axis lines, labels and tick labels visible on the dark
        # background; keep the type label small so 14 stacked rows do
        # not crowd
        chart.y_axis.label_size = 12
        for axis in (chart.x_axis, chart.y_axis):
            axis.pen.color = 'white'
            axis.GetTitleProperties().SetColor(1, 1, 1)
            axis.GetLabelProperties().SetColor(1, 1, 1)
        traces[cell_name] = []
        lines[cell_name] = chart.line(
            [t0, t1],
            [frame0[cell_name], frame1[cell_name]],
            color=celltype_attr[celltype]['color'],
            width=2.0,
        )
        charts.append(chart)
        plotter.add_chart(chart)

    # Seed the rolling buffers with the two frames already read
    for t, frame in ((t0, frame0), (t1, frame1)):
        times.append(t)
        for cell_name in traces:
            traces[cell_name].append(frame.get(cell_name, np.nan))

    def _refresh():
        window_t = [times[-1] - window, times[-1]]
        for cell_name, line in lines.items():
            line.update(list(times), traces[cell_name])
        for chart in charts:
            chart.x_range = window_t

    _refresh()

    def update(step):
        try:
            t, frame = next(data)
        except StopIteration:
            return
        for cell_name, vm in frame.items():
            actor = glyph_actors.get(cell_name)
            if actor is not None:
                actor.prop.color = vm_to_rgb(cell_name.partition('_')[0], vm)
        times.append(t)
        if len(times) > buflen:
            del times[0]
        for cell_name in reps.values():
            traces[cell_name].append(frame.get(cell_name, np.nan))
            if len(traces[cell_name]) > buflen:
                del traces[cell_name][0]
        _refresh()
        plotter.render()

    plotter.iren.initialize()
    plotter.add_timer_event(max_steps=10_000_000, duration=interval, callback=update)
    plotter.show()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        display_activity(sys.argv[1])
    else:
        fpath = '../../../traub_2005_full/dataviz/test_data/data_20111025_115951_4849.h5'
        display_data_2(fpath)

#
# vis.py ends here
