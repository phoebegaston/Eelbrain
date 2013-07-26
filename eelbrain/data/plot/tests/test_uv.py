# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import numpy as np
from matplotlib import pyplot as plt

from eelbrain.data import var, datasets, plot


def test_barplot():
    "Test plot.uv.barplot"
    ds = datasets.get_rand()
    plot.uv.barplot('Y', 'A%B', match='rm', ds=ds)
    plt.close('all')


def test_boxplot():
    "Test plot.uv.boxplot"
    ds = datasets.get_rand()
    plot.uv.boxplot('Y', 'A%B', match='rm', ds=ds)
    plt.close('all')


def test_histogram():
    "Test plot.uv.histogram"
    ds = datasets.get_rand()
    plot.uv.histogram('Y', 'A%B', ds=ds)
    plot.uv.histogram('Y', 'A%B', match='rm', ds=ds)
    plt.close('all')


def test_scatterplot():
    "Test plot.uv.corrplot and lot.uv.regplot"
    ds = datasets.get_rand()
    ds['cov'] = ds['Y'] + np.random.normal(0, 1, (60,))

    plot.uv.corrplot('Y', 'cov', ds=ds)
    plot.uv.corrplot('Y', 'cov', 'A%B', ds=ds)

    plot.uv.regplot('Y', 'cov', ds=ds)
    plot.uv.regplot('Y', 'cov', 'A%B', ds=ds)

    plt.close('all')


def test_timeplot():
    "Test plot.uv.timeplot"
    ds = datasets.get_rand()
    ds['seq'] = var(np.arange(2).repeat(30))
    plot.uv.timeplot('Y', 'B', 'seq', match='rm', ds=ds)
    plt.close('all')
