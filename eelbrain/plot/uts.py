'''
plotting functions for univariate uniform time-series.



Created on Mar 13, 2012

@author: christian
'''
from __future__ import division


import numpy as np
import scipy.stats
import matplotlib.cm as _cm
import matplotlib.pyplot as plt

from eelbrain.vessels.structure import celltable
from eelbrain.wxutils import mpl_canvas

import _base


__hide__ = ['plt', 'division', 'celltable']


def stat(Y='Y', X=None, dev=scipy.stats.sem, main=np.mean,
         sub=None, match=None, ds=None,
         figsize=(6,3), dpi=90, legend='upper right', title=True, ylabel=True,
         xdim='time', cm=_cm.jet):
    """
    Plots statistics for a one-dimensional ndvar
    
    Y : 1d-ndvar
        dependent variable (one-dimensional ndvar)
    X : categorial or None
        conditions which should be plotted separately
    main : func | None
        central tendency (function that takes an ``axis`` argument).
    dev : func | 'all' | float
        Measure for spread / deviation from the central tendency. Either a 
        function that takes an ``axis`` argument, 'all' to plot all traces, or
        a float to plot all traces with a certain alpha value
    ds : dataset
        if ``var`` or ``X`` is submitted as string, the ``ds`` argument 
        must provide a dataset containing those variables.
    
    **plotting parameters:**
    
    xdim : str
        dimension for the x-axis (default is 'time') 
    cm : matplotlib colormap
        colormap from which colors for different categories in ``X`` are 
        derived
    
    **figure parameters:**
    
    legend : str | None
        matplotlib figure legend location argument
    title : str | True | False
        axes title; if ``True``, use ``var.name``
    
    """
    ct = celltable(Y, X, sub=sub, match=match, ds=ds)
    
    if title is True:
        title = ct.Y.name
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    top = .9 # - bool(title) * .1
    plt.subplots_adjust(.1, .1, .95, top, .1, .4)
    ax = plt.axes()
    ax.x_fmt = "t = %.3f s"
        
    h = []
    legend_h = []
    legend_lbl = []
    N = len(ct)
    for i, cell in enumerate(ct.cells):
        lbl = ct.cell_label(cell, ' ')
        c = cm(i / N)
        _h = _plt_stat(ax, ct.data[cell], main, dev, label=lbl, xdim=xdim, color=c)
        h.append(_h)
        legend_h.append(_h['main'][0])
        legend_lbl.append(lbl)
    
    dim = ct.Y.get_dim(xdim)
    if title:
        ax.set_title(title)
    ax.set_xlabel(dim.name)
    ax.set_xlim(min(dim), max(dim))
    if legend and len(ct) > 1:
        fig.legend(legend_h, legend_lbl, loc=legend)
    
    if ylabel is True:
        ylabel = Y.properties.get('unit', None)
    
    if ylabel:
        ax.set_ylabel(ylabel)


    fig.tight_layout()
    fig.show()
    return fig


class clusters(mpl_canvas.CanvasFrame):
    def __init__(self, epochs, pmax=0.05, t=True,
                 title=None, cm=_cm.jet,
                 width=6, height=3, frame=.1, dpi=90,
                 overlay=False):
        """
        Specialized plotting function for Permutation Cluster test results
        
        t : bool
            plot threshold
        
        """
        epochs = self.epochs = _base.unpack_epochs_arg(epochs, 1)
        
        # create figure
        N = len(epochs)
        x_size = width
        y_size = height if overlay else height * N 
        figsize = (x_size, y_size)
        
        if title:
            title = unicode(title)
        else:
            title=''
        
        super(clusters, self).__init__(title=title, figsize=figsize, dpi=dpi)
        self.figure.subplots_adjust(hspace=.2, top=.95, bottom=.05)
        
        width = .85
        if overlay:
            height = .95
            ax = self.figure.add_subplot(111)
        else:
            height = .95 / N
        
        for i, layers in enumerate(epochs):
            if not overlay: # create axes
                ax = self.figure.add_subplot(N, 1, i + 1)
                ax.set_title(layers[0].name)
                        
            # color
            color = cm(i / N)
            _ax_clusters(ax, layers, color=color, pmax=pmax, t=t)
        
        self.Show()



def _ax_uts(ax, layers, color=None, xdim='time'):
    for l in layers:
        _plt_uts(ax, l, color=color, xdim=xdim)
    
    x = layers[0].get_dim(xdim).x
    ax.set_xlim(x[0], x[-1])


def _plt_uts(ax, layer, color=None, xdim='time'):
    x = layer.get_dim(xdim).x
    y = layer.get_data((xdim,))
    ax.plot(x, y, color=color)


def _ax_clusters(ax, layers, color=None, pmax=0.05, t=True, xdim='time'):
    if t is True:
        t = layers[0].properties.get('tF', None)
    if t:
        ax.axhline(t, color='k')
    
    _plt_uts(ax, layers[0], color=color, xdim=xdim)
    
    for l in layers[1:]:
        if l.properties['p'] <= pmax:
            _plt_cluster(ax, l, color=color, xdim=xdim, y=layers[0])
    
    x = layers[0].get_dim(xdim).x
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)


def _plt_cluster(ax, ndvar, color=None, y=None, xdim='time',
                 hatch='/'):
    x = ndvar.get_dim(xdim).x
    v = ndvar.get_data((xdim,))
    where = np.where(v)[0]
    assert np.abs(np.diff(where)).sum() <= 2
    x0 = where[0]
    x1 = where[-1]
    
    if y is None:
        ax.vspan(x0, x1, color=color, hatch=hatch, fill=False)
    else:
        y = y.get_data((xdim,))
        ax.fill_between(x, y, where=v, color=color, alpha=0.5)


def _plt_stat(ax, ndvar, main, dev, label=None, xdim='time', color=None, **kwargs):
    h = {}
    dim = ndvar.get_dim(xdim)
    x = dim.x
    y = ndvar.get_data(('case', 'time'))
    
    if color:
        kwargs['color'] = color
    
    main_kwargs = kwargs.copy()
    dev_kwargs = kwargs.copy()
    if label:
        main_kwargs['label'] = label
    
    if np.isscalar(dev):
        dev_kwargs['alpha'] = dev
        dev = 'all'
    else:
        dev_kwargs['alpha'] = .3
    
    if dev =='all':
        if 'linewidth' in kwargs:
            main_kwargs['linewidth'] = kwargs['linewidth'] * 2
        elif 'lw' in kwargs:
            main_kwargs['lw'] = kwargs['lw'] * 2
        else:
            main_kwargs['lw'] = 2
    
    # plot main
    if hasattr(main, '__call__'):
        y_ct = main(y, axis=0)
        h['main'] = ax.plot(x, y_ct, zorder=5, **main_kwargs)
    elif dev == 'all':
        pass
    else:
        raise ValueError("Invalid argument: main=%r" % main)
    
    # plot dev
    if hasattr(dev, '__call__'):
        ydev = dev(y, axis=0)
        h['dev'] = ax.fill_between(x, y_ct-ydev, y_ct+ydev, zorder=0, **dev_kwargs)
    elif dev == 'all':
        h['dev'] = ax.plot(x, y.T, **dev_kwargs)
        dev = None
    else:
        pass
    
    return h


