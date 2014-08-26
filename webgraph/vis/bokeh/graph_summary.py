#!/usr/bin/env python
"""graph_summary.py -- a set of plots summarizing a graph

USAGE:
    python graph_summary.py <view_option>

    view_option:  - static    A single view of the page <default>

AUTHOR:
    Andy R. Terrel <aterrel@continuum.io>

"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import sys
import time
from traceback import print_exc

import numpy as np
import pandas as pd

from bokeh.objects import (CategoricalAxis, ColumnDataSource, DataRange1d, 
                           HoverTool, FactorRange, Glyph, Grid, LinearAxis,
                           Plot, PanTool, WheelZoomTool, LogAxis)
from bokeh.glyphs import Rect, Text
import bokeh.glyphs as glyphs
import bokeh.plotting as plt


def make_box_violin_plot(data, maxwidth=0.9):
    """ 
    data: dict[Str -> List[Number]]
    maxwidth: float
        Maximum width of tornado plot within each factor/facet

    Returns the plot object 
    """
    print("Plotting box violin graph")
    plot_width = 500
    plot_height = 350
    df = pd.DataFrame(columns=["group", "width", "height", "texts", "cats"])
    bar_height = 50
    bins = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e10]
    # Compute histograms, while keeping track of max Y values and max counts
    for i, (group, vals) in enumerate(data.iteritems()):
        hist, edges = np.histogram(vals, bins)
        df = df.append(pd.DataFrame(dict(
            group = group,
            width = np.log2(hist[1:]),
            height = np.ones(len(hist) - 1),
            texts = ["%d Nodes" % i for i in hist[1:-1]] + ["%d" % hist[-1]],
            cats = [">10^%d" % np.log10(bin) for bin in bins[1:-1]],
            )))
            
    df.replace(-np.inf, 0)

    # Normalize the widths
    df["width"] *= (maxwidth / df["width"].max())
    
    ds = ColumnDataSource(df)

    xdr = FactorRange(factors=sorted(df["group"].unique()))
    ydr = FactorRange(factors=list(df["cats"]))

    plot = Plot(data_sources=[ds], x_range=xdr, y_range=ydr,
                title="Degree Distribution (log scale)",
                plot_width=plot_width, plot_height=plot_height, 
                tools=[])
    yaxis = CategoricalAxis(plot=plot, location="left", axis_label="degree")
    plot.left.append(yaxis)
    
    glyph = Rect(x="group", y="cats", width="width", height="height",
                 fill_color="#3366ff")
    text_glyph = Text(x="group", y="cats", text="texts", text_baseline="middle",
                      text_align="center", angle=0)
    plot.renderers.append(Glyph(data_source=ds, xdata_range=xdr, ydata_range=ydr,
                                 glyph=glyph))
    plot.renderers.append(Glyph(data_source=ds, xdata_range=xdr, ydata_range=ydr,
                                glyph=text_glyph))
    return plot


def find_in_array(array, val):
    for i, arr_val in enumerate(array):
        if arr_val == val:
            return i
    return -1


def build_subgraph(nodes, csr_offsets, csr_indices):
    print("Building subgraph of %d nodes" % len(nodes))
    row_offsets = np.zeros(len(nodes) + 1, dtype=int)
    col_indices = np.empty(csr_indices.shape, dtype=int)
    curr_edge = 0
    for new_id, node_id in enumerate(nodes):
        cols = csr_indices[csr_offsets[node_id]:csr_offsets[node_id+1]]
        row_offsets[new_id] = curr_edge
        for col_id in cols:
            col_new_id = find_in_array(nodes, col_id)
            if col_new_id != -1:
                col_indices[curr_edge] = col_new_id
                curr_edge += 1
    row_offsets[new_id+1] = curr_edge
    return row_offsets, col_indices


def compute_adj(rows, cols):
    N = len(rows) - 1
    M = rows[N]
    sparcity = (M / (N*N)) * 100
    print("Computing adjaency of %d nodes, %d edges, %.1f%% sparsity" \
          % (N, M, sparcity))
    adj = np.zeros((N, N))
    for i in range(N):
        start, end = rows[i], rows[i+1]
        for j in range(start, end):
            adj[N-i-1, cols[j]] = 100
    return adj, sparcity


def plot_adj(adj, sparcity):
    N, _ = adj.shape
    TOOLS="pan,wheel_zoom,box_zoom,reset,click,previewsave"
    plt.figure(plot_width=N+100, plot_height=N+100, tools=TOOLS)
    plt.image(image=[adj], x =[0], y=[0], dw=[N], dh=[N], 
              x_range=[0, N], y_range=[0, N],
              palette=["Blues-3"],
              title="Adjacency of top %d nodes (%.2f%% sparcity)" % (N, sparcity),
              x_axis_type=None, y_axis_type=None)
    return plt.curplot()


def plot_adjacency_graph(nodes, csr_offsets, csr_indices, max_nodes=800):
    print("Plotting adjacency graph")
    rows, cols = build_subgraph(nodes[:max_nodes], csr_offsets, csr_indices)
    adj, sparcity = compute_adj(rows, cols)
    return plot_adj(adj, sparcity)


def plot_circle_density(nodes, degrees, plot_width=800, plot_height=800):
    print("Plotting circle density graph")
    TOOLS="hover,pan,wheel_zoom,box_zoom,reset,click,previewsave"    
    plt.figure(plot_width=plot_width, plot_height=plot_height, tools=TOOLS)
    theta = np.random.uniform(0, 2*np.pi, size=len(nodes))
    max_d, min_d = np.max(degrees), np.min(degrees)
    scale = 1.0/np.log(degrees) - 1.0/np.log(max_d)
    xs = np.cos(theta)*scale
    ys = np.sin(theta)*scale
    source_dict = dict(
        xs = xs,
        ys = ys,
        degrees = degrees,
        nodes = nodes,
        alphas = np.log(degrees)/np.log(max(degrees)),
    )
    source = ColumnDataSource(source_dict)
    plt.hold(True)
    plt.circle('xs', 'ys', source=source,
               radius=0.0025,
               fill_alpha='alphas',
               x_axis_type=None, y_axis_type=None, 
               title="Density Distribution of Degrees")
    plt.text([max(xs), max(xs)], 
             [.95*max(ys), .85*max(ys)], 
             ["distance from center = 1 / log(deg)",
              "angle = random"], 
             angle=0,
             text_baseline="bottom", text_align="right")
    hover = [t for t in plt.curplot().tools if isinstance(t, HoverTool)][0]
    hover.tooltips = OrderedDict([
        ('node', '@nodes'), ('degree', '@degrees')
    ])
    plt.hold(False)
    return plt.curplot()


def get_data():
    """Dummy for testing"""
    degrees = np.load("data/degrees.npy")
    nodes = np.load("data/top_nodes.npy")
    csr_offsets = np.load("data/csr_offsets.npy")
    csr_indices = np.load("data/csr_indices.npy")
    return degrees, nodes, csr_offsets, csr_indices


def plot_graphs(nodes, degrees, csr_offsets, csr_indices):
    data = {"": degrees}
    doc = plt.curdoc()

    plt.output_file("graph_summaries.html", title="Graph Summary")
    p = make_box_violin_plot(data)
    doc.add(p)
    doc._current_plot = p
    plot_adjacency_graph(nodes, csr_offsets, csr_indices)
    plot_circle_density(nodes, degrees)
    plt.show()


if __name__ == "__main__":
    plot_graphs(*get_data())


    

