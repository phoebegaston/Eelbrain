"""GUI for rejecting epochs"""

# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from __future__ import division

import math
import os
import time

import mne
import numpy as np
import wx

from .. import load, save, plot
from .._data_obj import Dataset, Factor, Var, corr, asndvar
from ..plot._base import find_fig_vlims
from ..plot.utsnd import _ax_bfly_epoch
from ..plot.nuts import _plt_bin_nuts
from ..wxutils import Icon, ID, logger, REValidator
from ..wxutils.mpl_canvas import FigureCanvasPanel
from .history import History


class ChangeAction():
    """Action objects are kept in the history and can do and undo themselves"""

    def __init__(self, desc, index, old_accept, new_accept, old_tag, new_tag,
                 old_path=None, new_path=None):
        """
        Parameters
        ----------
        desc : str
            Description of the action
            list of (i, name, old, new) tuples
        """
        self.desc = desc
        self.index = index
        self.old_path = old_path
        self.old_accept = old_accept
        self.old_tag = old_tag
        self.new_path = new_path
        self.new_accept = new_accept
        self.new_tag = new_tag

    def do(self, doc):
        doc.set_case(self.index, self.new_accept, self.new_tag)
        if self.new_path is not None:
            doc.set_path(self.new_path)

    def undo(self, doc):
        doc.set_case(self.index, self.old_accept, self.old_tag)
        if self.new_path is not None and self.old_path is not None:
            doc.set_path(self.old_path)


class Document(object):
    "Represents data for the current state of the Document"

    def __init__(self, ds, data='meg', accept='accept', blink='blink',
                 tag='rej_tag', trigger='trigger', path=None, bad_chs=None):
        """
        Parameters
        ----------
        ds : Dataset
            Dataset containing
            ...
        path : None | str
            Default location of the epoch selection file (used for save
            command). If the file exists, it is loaded as initial state.
        """
        if isinstance(ds, mne.Epochs):
            epochs = ds
            if not epochs.preload:
                err = ("Need Epochs with preloaded data (preload=True)")
                raise ValueError(err)
            ds = Dataset()
            ds[data] = epochs
            ds['trigger'] = Var(epochs.events[:, 2])

        data = asndvar(data, ds=ds)
        self.n_epochs = n = len(data)

        if not isinstance(accept, basestring):
            raise TypeError("accept needs to be a string")
        if accept not in ds:
            x = np.ones(n, dtype=bool)
            ds[accept] = Var(x)
        accept = ds[accept]

        if not isinstance(tag, basestring):
            raise TypeError("tag needs to be a string")
        if tag in ds:
            tag = ds[tag]
        else:
            tag = Factor([''], rep=n, name=tag)
            ds.add(tag)

        if not isinstance(trigger, basestring):
            raise TypeError("trigger needs to be a string")
        if trigger in ds:
            trigger = ds[trigger]
        else:
            err = ("ds does not contain a variable named %r. The trigger "
                   "parameters needs to point to a variable in ds containing "
                   "trigger values." % trigger)
            raise KeyError(err)

        if isinstance(blink, basestring):
            if ds is not None:
                blink = ds.get(blink, None)
        elif blink == True:
            if 'edf' in ds.info:
                tmin = data.time.tmin
                tmax = data.time.tmax
                _, blink = load.eyelink.artifact_epochs(ds, tmin, tmax,
                                                        esacc=False)
            else:
                msg = ("No eye tracker data was found in ds.info['edf']. Use "
                       "load.eyelink.add_edf(ds) to add an eye tracker file "
                       "to a Dataset ds.")
                wx.MessageBox(msg, "Eye Tracker Data Not Found")
                blink = None
        elif blink is not None:
            raise TypeError("blink needs to be a string or None")

        # data
        self.data = data
        self.data_good = data  # with bad channels removed
        self.accept = accept
        self.tag = tag
        self.trigger = trigger
        self.blink = blink

        self._good_chs = slice(None)
        self._bad_chs = []

        # publisher
        self._case_change_subscriptions = []
        self._path_change_subscriptions = []

        # finalize
        if bad_chs is not None:
            self._set_bad_chs(bad_chs, reset=True)

        self.saved = True  # managed by the history
        self.path = path
        if path and os.path.exists(path):
            accept, tag = self.read_rej_file(path)
            self.accept[:] = accept
            self.tag[:] = tag

    def get_bad_chs(self, name=True):
        """Get the channels currently set as bad

        Parameters
        ----------
        name : bool
            Return channel names (otherwise the channel index is returned).

        Returns
        -------
        bad_chs : None | list of int, str
            Channels currenty excluded.
        """
        if name:
            return [self.doc.data.sensor.names[i] for i in self._bad_chs]
        else:
            return self._bad_chs[:]

    def get_epoch(self, i):
        name = 'Epoch %i' % i
        epoch = self.data.sub(case=i, name=name)
        return epoch

    def get_grand_average(self):
        "Grand average of all accepted epochs"
        out = self.data[self.accept].mean('case')
        out.name = "Grand Average"
        return out

    def set_bad_chs(self, bad_chs, reset=False):
        """Set the channels to treat as bad (i.e., exclude)

        Parameters
        ----------
        bad_chs : None | list of str, int
            List of channels to treat as bad (as name or index).
        reset : bool
            Reset previously set bad channels to good.
        """
        if reset:
            del self._bad_chs[:]
            self.data_good = self.data

        if bad_chs is None:
            return

        bad_chs = self.doc.data.sensor.dimindex(bad_chs)
        for ch in bad_chs:
            if ch in self._bad_chs:
                continue
            self._bad_chs.append(ch)
        good_chs = self.data.sensor.isnotin(self._bad_chs)
        self.data_good = self.data.sub(sensor=good_chs)

    def set_case(self, index, state, tag=None):
        self.accept[index] = state
        if tag is not None:
            self.tag[index] = tag

        for func in self._case_change_subscriptions:
            func(index)

    def set_path(self, path):
        """Set the path

        Parameters
        ----------
        path : str
            Path under which to save. The extension determines the way file
            (*.pickled -> pickled Dataset; *.txt -> tsv)
        """
        root, ext = os.path.splitext(path)
        if ext == '':
            path = root + '.txt'

        self.path = path
        for func in self._path_change_subscriptions:
            func()

    def read_rej_file(self, path):
        "Read a file making sure it is compatible"
        _, ext = os.path.splitext(path)
        if ext == '.pickled':
            ds = load.unpickle(path)
        elif ext == '.txt':
            ds = load.tsv(path, delimiter='\t')
        else:
            raise ValueError("Unknown file extension for rejections: %r" % ext)

        # check file
        if ds.n_cases != self.n_epochs:
            raise RuntimeError("Unequal number of cases")
        elif not np.all(ds[self.trigger.name] == self.trigger):
            raise RuntimeError("Trigger mismatch")

        accept = ds['accept']
        if 'rej_tag' in ds:
            tag = ds['rej_tag']
        else:
            tag = Factor([''], rep=self.n_epochs, name='rej_tag')

        return accept, tag

    def save(self):
        # find dest path
        _, ext = os.path.splitext(self.path)

        # create Dataset to save
        info = {'bad_chs': self.get_bad_chs()}
        ds = Dataset(self.trigger, self.accept, self.tag, info=info)

        if ext == '.pickled':
            save.pickle(ds, self.path)
        elif ext == '.txt':
            ds.save_txt(self.path)
        else:
            raise ValueError("Unsupported extension: %r" % ext)

    def subscribe_to_case_change(self, callback):
        "callback(index)"
        self._case_change_subscriptions.append(callback)

    def subscribe_to_path_change(self, callback):
        "callback(path)"
        self._path_change_subscriptions.append(callback)



class Model(object):
    """Manages a document as well as its history"""

    def __init__(self, doc):
        self.doc = doc
        self.history = History(doc)

    def auto_reject(self, threshold=2e-12, method='abs', above=False,
                    below=True):
        """
        Marks epochs based on a threshold criterion

        Parameters
        ----------
        threshold : scalar
            The threshold value. Examples: 1.25e-11 to detect saturated
            channels; 2e-12: for conservative MEG rejection.
        method : 'abs' | 'p2p'
            How to apply the threshold. With "abs", the threshold is applied to
            absolute values. With 'p2p' the threshold is applied to
            peak-to-peak values.
        above, below: True, False, None
            How to mark segments that do (above) or do not (below) exceed the
            threshold: True->good; False->bad; None->don't change
        """
        args = ', '.join(map(str, (threshold, method, above, below)))
        logger.info("Auto-reject trials: %s" % args)

        x = self.doc.data_good
        if method == 'abs':
            x_max = x.abs().max(('time', 'sensor'))
            sub_threshold = x_max <= threshold
        elif method == 'p2p':
            p2p = x.max('time') - x.min('time')
            max_p2p = p2p.max('sensor')
            sub_threshold = max_p2p <= threshold
        else:
            raise ValueError("Invalid method: %r" % method)

        accept = sub_threshold.copy()

        if below is False:
            accept[sub_threshold] = False
        elif below is None:
            accept[sub_threshold] = self.doc.accept.x[sub_threshold]
        elif below is not True:
            err = "below needs to be True, False or None, got %s" % repr(below)
            raise TypeError(err)

        if above is True:
            accept[sub_threshold == False] = True
        elif above is None:
            accept = np.where(sub_threshold, accept, self.doc.accept.x)
        elif above is not False:
            err = "above needs to be True, False or None, got %s" % repr(above)
            raise TypeError(err)

        index = np.where(self.doc.accept.x != accept)[0]
        old_accept = self.doc.accept[index]
        new_accept = accept[index]
        old_tag = self.doc.tag[index]
        new_tag = "%s_%s" % (method, threshold)
        desc = "Threshold-%s" % method
        action = ChangeAction(desc, index, old_accept, new_accept, old_tag,
                              new_tag)
        logger.info("Auto-rejecting %i trials based on threshold %s" %
                    (len(index), method))
        self.history.do(action)

    def clear(self):
        desc = "Clear"
        index = np.logical_not(self.doc.accept.x)
        old_tag = self.doc.tag[index]
        action = ChangeAction(desc, index, False, True, old_tag, 'clear')
        logger.info("Clearing %i rejections" % index.sum())
        self.history.do(action)

    def load(self, path):
        new_accept, new_tag = self.doc.read_rej_file(path)

        # create load action
        desc = "Load File"
        new_path = path
        index = slice(None)
        old_accept = self.doc.accept
        old_tag = self.doc.tag
        old_path = self.doc.path
        action = ChangeAction(desc, index, old_accept, new_accept, old_tag,
                              new_tag, old_path, new_path)
        self.history.do(action)
        self.history.register_save()

    def save(self):
        self.doc.save()
        self.history.register_save()

    def save_as(self, path):
        self.doc.set_path(path)
        self.save()

    def set_case(self, i, state, tag=None, desc="Manual Change"):
        old_accept = self.doc.accept[i]
        if tag is None:
            old_tag = None
        else:
            old_tag = self.doc.tag[i]
        action = ChangeAction(desc, i, old_accept, state, old_tag, tag)
        self.history.do(action)


class Frame(wx.Frame):  # control
    "View object of the epoch selection GUI"

    def __init__(self, parent, model, config, nplots, topo, mean, vlim,
                 plot_range, color, lw, mark, mcolor, mlw, antialiased, pos,
                 size):
        """View object of the epoch selection GUI

        Parameters
        ----------
        parent : wx.Frame
            Parent window.
        others :
            See TerminalInterface constructor.
        """
        if pos is None:
            pos = (config.ReadInt("pos_horizontal", -1),
                   config.ReadInt("pos_vertical", -1))
        else:
            pos_h, pos_v = pos
            config.WriteInt("pos_horizontal", pos_h)
            config.WriteInt("pos_vertical", pos_v)
            config.Flush()

        if size is None:
            size = (config.ReadInt("size_width", 800),
                   config.ReadInt("size_height", 600))
        else:
            w, h = pos
            config.WriteInt("size_width", w)
            config.WriteInt("size_height", h)
            config.Flush()

        super(Frame, self).__init__(parent, -1, "Select Epochs", pos, size)

        # bind close event to save window properties in config
        self.Bind(wx.EVT_CLOSE, self.OnClose, self)

        self.config = config
        self.model = model
        self.doc = model.doc
        self.history = model.history

#         self._topo_fig = None
        self._saved = True

        # setup figure canvas
        self.canvas = FigureCanvasPanel(self)
        self.figure = self.canvas.figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05,
                                    top=.95, hspace=.5)
        self.canvas.mpl_connect('motion_notify_event', self.OnPointerMotion)
        self.canvas.mpl_connect('axes_leave_event', self.OnPointerLeaveAxes)

        # Toolbar
        tb = self.CreateToolBar(wx.TB_HORIZONTAL)
        tb.SetToolBitmapSize(size=(32, 32))

        if hasattr(parent, 'shell') and hasattr(parent.shell, 'attach'):
            tb.AddLabelTool(ID.ATTACH, "Attach", Icon("actions/attach"))

        tb.AddLabelTool(wx.ID_SAVE, "Save",
                        Icon("tango/actions/document-save"), shortHelp="Save")
        tb.AddLabelTool(wx.ID_SAVEAS, "Save As",
                        Icon("tango/actions/document-save-as"),
                        shortHelp="Save As")
        tb.AddLabelTool(wx.ID_OPEN, "Load",
                        Icon("tango/actions/document-open"),
                        shortHelp="Open Rejections")

        tb.AddLabelTool(ID.UNDO, "Undo", Icon("tango/actions/edit-undo"),
                        shortHelp="Undo")
        tb.AddLabelTool(ID.REDO, "Redo", Icon("tango/actions/edit-redo"),
                        shortHelp="Redo")
        tb.AddSeparator()

        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        self.page_choice = wx.Choice(tb, -1)
        tb.AddControl(self.page_choice)
        tb.Bind(wx.EVT_CHOICE, self.OnPageChoice)

        # --> forward / backward
        self.back_button = tb.AddLabelTool(wx.ID_BACKWARD, "Back",
                                           Icon("tango/actions/go-previous"))
        self.next_button = tb.AddLabelTool(wx.ID_FORWARD, "Next",
                                           Icon("tango/actions/go-next"))
        tb.AddSeparator()

        # --> Thresholding
        self.threshold_button = wx.Button(tb, ID.THRESHOLD, "Threshold")
        tb.AddControl(self.threshold_button)

        # exclude channels
#         btn = wx.Button(tb, ID.EXCLUDE_CHANNELS, "Exclude Channel")
#         tb.AddControl(btn)
#         btn.Bind(wx.EVT_BUTTON, self.OnExcludeChannel)

        # right-most part
        if wx.__version__ >= '2.9':
            tb.AddStretchableSpace()
        else:
            tb.AddSeparator()

#         tb.AddLabelTool(wx.ID_HELP, 'Help', Icon("tango/apps/help-browser"))
#         self.Bind(wx.EVT_TOOL, self.OnHelp, id=wx.ID_HELP)

#         tb.AddLabelTool(ID.FULLSCREEN, "Fullscreen",
#                         Icon("tango/actions/view-fullscreen"))
#         self.Bind(wx.EVT_TOOL, self.OnShowFullScreen, id=ID.FULLSCREEN)

        # Grand-average plot
        self.grand_av_button = wx.Button(tb, ID.GRAND_AVERAGE, "GA")
        self.grand_av_button.SetHelpText("Plot the grand average of all "
                                         "accepted epochs")
        tb.AddControl(self.grand_av_button)

        tb.Realize()

        self.CreateStatusBar()

        # setup plot parameters
        self._vlims = find_fig_vlims([[self.doc.data]])
        if vlim is not None:
            for k in self._vlims:
                self._vlims[k] = (-vlim, vlim)
        self._bfly_kwargs = {'plot_range': plot_range, 'color': color, 'lw': lw,
                             'mcolor': mcolor, 'mlw': mlw,
                             'antialiased': antialiased, 'vlims': self._vlims}
        self._topo_kwargs = {'vlims': self._vlims, 'title': None}
        self._SetPlotStyle(mark=mark)
        self._SetLayout(nplots, topo, mean)

        # Finalize
        self.ShowPage(0)
        self.UpdateTitle()

    def _create_menu(self):
        # Menu
        m = self.fileMenu = wx.Menu()
        m.Append(wx.ID_OPEN, '&Open... \tCtrl+O', 'Open file')
#         m.Append(wx.ID_REVERT, '&Revert', 'Revert to the last saved version')
        m.AppendSeparator()
        m.Append(wx.ID_CLOSE, '&Close \tCtrl+W', 'Close Window')
        m.Append(wx.ID_SAVE, '&Save \tCtrl+S', 'Save file')
        m.Append(wx.ID_SAVEAS, 'Save &As... \tCtrl+Shift+S', 'Save file with '
                 'new name')
#         m.Append(ID.SAVEACOPY, 'Save A Cop&y', 'Save a copy of the file '
#                  'without changing the current file')

        # Edit
        m = self.editMenu = wx.Menu()
        m.Append(ID.UNDO, '&Undo \tCtrl+Z', 'Undo the last action')
        m.Append(ID.REDO, '&Redo \tCtrl+Shift+Z', 'Redo the last undone '
                 'action')
        m.AppendSeparator()
        m.Append(wx.ID_CLEAR, 'Cle&ar', 'Select all epochs')

        # View
        m = self.viewMenu = wx.Menu()
        m.Append(ID.SET_VLIM, "Set Y-Axis Limit... \tCtrl+l", "Change the Y-"
                 "axis limit in epoch plots")
        m.Append(ID.SET_LAYOUT, "&Set Layout... \tCtrl+Shift+l", "Change the "
                 "page layout")
        m.AppendCheckItem(ID.PLOT_RANGE, "&Plot Data Range \tCtrl+r", "Plot "
                          "data range instead of individual sensor traces")
#         m.Append(wx.ID_TOGGLE_MAXIMIZE, '&Toggle Maximize\tF11', 'Maximize/'
#                  'Restore Application')

        # Go
        m = self.goMenu = wx.Menu()
        m.Append(wx.ID_FORWARD, '&Forward \tCtrl+]', 'Go One Page Forward')
        m.Append(wx.ID_BACKWARD, '&Back \tCtrl+[', 'Go One Page Back')

#         m = self.helpMenu = wx.Menu()
#         m.Append(wx.ID_HELP, '&Help\tF1', 'Help!')
#         m.AppendSeparator()
#         m.Append(wx.ID_ABOUT, '&About...', 'About this program')

        b = wx.MenuBar()
        b.Append(self.fileMenu, '&File')
        b.Append(self.editMenu, '&Edit')
        b.Append(self.viewMenu, '&View')
        b.Append(self.goMenu, '&Go')
#         b.Append(self.helpMenu, '&Help')
#         self.menuBar = b
        self.SetMenuBar(b)

    def CaseChanged(self, index):
        "updates the states of the segments on the current page"
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or self.doc.n_epochs
            index = xrange(start, stop)
        elif index.dtype.kind == 'b':
            index = np.nonzero(index)[0]

        # update epoch plots
        axes = []
        for idx in index:
            if idx in self._axes_by_idx:
                ax = self._axes_by_idx[idx]
                state = self.doc.accept[idx]
                ax_idx = ax.ax_idx
                h = self._case_plots[ax_idx]
                h.set_state(state)
                axes.append(ax)

        # update mean plot
        if self._plot_mean:
            mseg = self._get_page_mean_seg()
            self._mean_plot.set_data(mseg)
            axes.append(self._mean_ax)

        self.canvas.redraw(axes=axes)

    def OnClose(self, event):
        logger.debug("Frame.OnClose(), saving window properties...")
        pos_h, pos_v = self.GetPosition()
        w, h = self.GetSize()

        self.config.WriteInt("pos_horizontal", pos_h)
        self.config.WriteInt("pos_vertical", pos_v)
        self.config.WriteInt("size_width", w)
        self.config.WriteInt("size_height", h)
        self.config.Flush()

        event.Skip()

    def OnPageChoice(self, event):
        "called by the page Choice control"
        page = event.GetSelection()
        self.ShowPage(page)

    def OnPointerLeaveAxes(self, event):
        sb = self.GetStatusBar()
        sb.SetStatusText("", 0)

    def OnPointerMotion(self, event):
        "update view on mouse pointer movement"
        ax = event.inaxes
        if not ax:
            self.SetStatusText("")
            return

        # compose status text
        y_fmt = getattr(ax, 'y_fmt', 'y = %.3g')
        x_fmt = getattr(ax, 'x_fmt', 'x = %.3g')
        y_txt = y_fmt % event.ydata
        x_txt = x_fmt % event.xdata
        pos_txt = ',  '.join((x_txt, y_txt))
        if ax.ax_idx >= 0:  # single trial plot
            txt = 'Epoch %i,   %%s' % ax.epoch_idx
        elif  ax.ax_idx == -1:  # mean plot
            txt = "Page average,   %s"
        else:
            txt = '%s'
        self.SetStatusText(txt % pos_txt)

        # update topomap
        if self._plot_topo and ax.ax_idx > -2:  # topomap ax_idx is -2
            t = event.xdata
            tseg = self._get_ax_data(ax.ax_idx, t)
            plot.topo._ax_topomap(self._topo_ax, [tseg], **self._topo_kwargs)
            self.canvas.redraw(axes=[self._topo_ax])

    def PlotCorrelation(self, ax_index):
        if ax_index == -1:
            seg = self._mean_seg
            name = 'Page Mean Neighbor Correlation'
        else:
            epoch_idx = self._epoch_idxs[ax_index]
            seg = self._case_segs[ax_index]
            name = 'Epoch %i Neighbor Correlation' % epoch_idx
        cseg = corr(seg, name=name)
        plot.Topomap(cseg, sensors='name')

    def PlotButterfly(self, ax_index):
        epoch = self._get_ax_data(ax_index)
        plot.TopoButterfly(epoch, vmax=self._vlims)

    def PlotGrandAverage(self):
        epoch = self.doc.get_grand_average()
        plot.TopoButterfly(epoch)

    def PlotTopomap(self, ax_index, time):
        tseg = self._get_ax_data(ax_index, time)
        plot.Topomap(tseg, sensors='name', vmax=self._vlims)

    def SetLayout(self, nplots=(6, 6), topo=True, mean=True):
        """Determine the layout of the Epochs canvas

        Parameters
        ----------
        nplots : int | tuple of 2 int
            Number of epoch plots per page. Can be an ``int`` to produce a
            square layout with that many epochs, or an ``(n_rows, n_columns)``
            tuple.
        topo : bool
            Show a topomap plot of the time point under the mouse cursor.
        mean : bool
            Show a plot of the page mean at the bottom right of the page.
        """
        self._SetLayout(nplots, topo, mean)
        self.ShowPage(0)

    def _SetLayout(self, nplots, topo, mean):
        if topo is None:
            topo = self.config.ReadBool('Layout/show_topo', True)
        else:
            topo = bool(topo)
            self.config.WriteBool('Layout/show_topo', topo)

        if mean is None:
            mean = self.config.ReadBool('Layout/show_mean', True)
        else:
            mean = bool(mean)
            self.config.WriteBool('Layout/show_mean', mean)

        if nplots is None:
            nrow = self.config.ReadInt('Layout/n_rows', 6)
            ncol = self.config.ReadInt('Layout/n_cols', 6)
            nax = ncol * nrow
            n_per_page = nax - bool(topo) - bool(mean)
        else:
            if isinstance(nplots, int):
                if nplots == 1:
                    mean = False
                elif nplots < 1:
                    raise ValueError("nplots needs to be >= 1; got %r" % nplots)
                nax = nplots + bool(mean) + bool(topo)
                nrow = math.ceil(math.sqrt(nax))
                ncol = int(math.ceil(nax / nrow))
                nrow = int(nrow)
                n_per_page = nplots
            else:
                nrow, ncol = nplots
                nax = ncol * nrow
                if nax == 1:
                    mean = False
                    topo = False
                elif nax == 2:
                    mean = False
                elif nax < 1:
                    err = ("nplots=%s: Need at least one plot." % str(nplots))
                    raise ValueError(err)
                n_per_page = nax - bool(topo) - bool(mean)
            self.config.WriteInt('Layout/n_rows', nrow)
            self.config.WriteInt('Layout/n_cols', ncol)
        self.config.Flush()

        self._plot_mean = mean
        self._plot_topo = topo

        # prepare segments
        n = self.doc.n_epochs
        self._nplots = (nrow, ncol)
        self._n_per_page = n_per_page
        self._n_pages = n_pages = int(math.ceil(n / n_per_page))

        # get a list of IDS for each page
        self._segs_by_page = []
        for i in xrange(n_pages):
            start = i * n_per_page
            stop = min((i + 1) * n_per_page, n)
            self._segs_by_page.append(range(start, stop))

        # update page selector
        pages = []
        for i in xrange(n_pages):
            istart = self._segs_by_page[i][0]
            if i == n_pages - 1:
                pages.append('%i: %i..%i' % (i, istart, self.doc.n_epochs))
            else:
                pages.append('%i: %i...' % (i, istart))
        self.page_choice.SetItems(pages)

    def SetPlotStyle(self, **kwargs):
        """Select channels to mark in the butterfly plots.

        Parameters
        ----------
        plot_range : bool
            In the epoch plots, plot the range of the data (instead of plotting
            all sensor traces). This makes drawing of pages quicker, especially
            for data with many sensors (default ``True``).
        color : None | matplotlib color
            Color for primary data (default is black).
        lw : scalar
            Linewidth for normal sensor plots.
        mark : None | index for sensor dim
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        mlw : scalar
            Line width for marked sensor plots.
        antialiased : bool
            Perform Antialiasing on epoch plots (associated with a minor speed
            cost).
        """
        self._SetPlotStyle(**kwargs)
        self.ShowPage()

    def _SetPlotStyle(self, **kwargs):
        "See .SetPlotStyle()"
        bf_kwargs = self._bfly_kwargs

        for key, value in kwargs.iteritems():
            if key == 'vlims':
                err = ("%r is an invalid keyword argument for this function"
                       % key)
                raise TypeError(err)
            elif key == 'mark':
                if value is None:
                    bf_kwargs['mark'] = None
                else:
                    bf_kwargs['mark'] = self.doc.data.sensor.dimindex(value)
            elif key in bf_kwargs:
                bf_kwargs[key] = value

        # update which traces to plot
        plot_range = bf_kwargs['plot_range']
        mark = bf_kwargs['mark']
        if plot_range or mark is None:
            traces = not bool(plot_range)
        else:
            traces = np.setdiff1d(np.arange(len(self.doc.data.sensor)), mark)
        bf_kwargs['traces'] = traces

    def SetVLim(self, vlim):
        """Set the value limits (butterfly plot y axes and topomap colormaps)

        Parameters
        ----------
        vlim : scalar | (scalar, scalar)
            For symmetric limits the positive vmax, for asymmetric limits a
            (vmin, vmax) tuple.
        """
        for p in self._case_plots:
            p.set_ylim(vlim)
        self._mean_plot.set_ylim(vlim)
        if np.isscalar(vlim):
            vlim = (-vlim, vlim)
        for key in self._vlims:
            self._vlims[key] = vlim
        self.canvas.draw()

    def ShowPage(self, page=None):
        "Dislay a specific page (start counting with 0)"
        wx.BeginBusyCursor()
        t0 = time.time()
        if page is None:
            page = self._current_page_i
        else:
            self._current_page_i = page
            self.page_choice.Select(page)

        self.figure.clf()
        nrow, ncol = self._nplots
        self._epoch_idxs = self._segs_by_page[page]

        # segment plots
        self._case_plots = []
        self._case_axes = []
        self._case_segs = []
        self._axes_by_idx = {}
        for i, epoch_idx in enumerate(self._epoch_idxs):
            name = 'Epoch %i' % epoch_idx
            case = self.doc.data.sub(case=epoch_idx, name=name)
            state = self.doc.accept[epoch_idx]
            ax = self.figure.add_subplot(nrow, ncol, i + 1, xticks=[0],
                                         yticks=[])
            h = _ax_bfly_epoch(ax, case, xlabel=None, ylabel=None, state=state,
                               **self._bfly_kwargs)
            if self.doc.blink is not None:
                _plt_bin_nuts(ax, self.doc.blink[epoch_idx],
                              color=(0.99, 0.76, 0.21))

            ax.ax_idx = i
            ax.epoch_idx = epoch_idx
            self._case_plots.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case)
            self._axes_by_idx[epoch_idx] = ax


        # mean plot
        if self._plot_mean:
            plot_i = nrow * ncol
            ax = self._mean_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            ax.ax_idx = -1

            mseg = self._mean_seg = self._get_page_mean_seg()
            self._mean_plot = _ax_bfly_epoch(ax, mseg, **self._bfly_kwargs)

        # topomap
        if self._plot_topo:
            plot_i = nrow * ncol - self._plot_mean
            ax = self._topo_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            ax.ax_idx = -2
            ax.set_axis_off()

        self.canvas.draw()
        self.canvas.store_canvas()

        dt = time.time() - t0
        logger.debug('Page draw took %.1f seconds.', dt)
        wx.EndBusyCursor()

    def UpdateTitle(self):
        if self.doc.path:
            title = os.path.basename(self.doc.path)
            if not self.doc.saved:
                title = '* ' + title
        else:
            title = 'Unsaved'

        self.SetTitle(title)

    def _get_page_mean_seg(self, sensor=None):
        page_segments = self._segs_by_page[self._current_page_i]
        page_index = np.zeros(self.doc.n_epochs, dtype=bool)
        page_index[page_segments] = True
        index = np.logical_and(page_index, self.doc.accept.x)
        mseg = self.doc.data.summary(case=index)
        if sensor is not None:
            mseg = mseg.sub(sensor=sensor)
        return mseg

    def _get_ax_data(self, ax_index, time=None):
        if ax_index == -1:
            seg = self._mean_seg
            epoch_name = 'Page Average'
        elif ax_index >= 0:
            epoch_idx = self._epoch_idxs[ax_index]
            epoch_name = 'Epoch %i' % epoch_idx
            seg = self._case_segs[ax_index]
        else:
            raise ValueError("ax_index needs to be >= -1, not %s" % ax_index)

        if time is not None:
            name = '%s, %.3f s' % (epoch_name, time)
            seg = seg.sub(time=time, name=name)

        return seg


class Controller(object):

    def __init__(self, parent, model, nplots, topo, mean, vlim, plot_range,
                 color, lw, mark, mcolor, mlw, antialiased, pos, size):
        """Controller object for SelectEpochs GUI

        Parameters
        ----------
        parent : wx.Frame
            Parent window.
        model : Model
            Document model.
        others :
            See TerminalInterface constructor.
        """
        self.config = wx.Config("Eelbrain")
        self.config.SetPath("SelectEpochs")
        self.frame = Frame(parent, model, self.config, nplots, topo, mean, vlim,
                           plot_range, color, lw, mark, mcolor, mlw,
                           antialiased, pos, size)
        self.model = model
        self.doc = model.doc
        self.history = model.history

        self.doc.subscribe_to_case_change(self.CaseChanged)
        self.doc.subscribe_to_path_change(self.frame.UpdateTitle)
        self.history.subscribe_to_saved_change(self.frame.UpdateTitle)

        f = self.frame
        f.Bind(wx.EVT_CLOSE, self.OnCloseEvent)
        f.Bind(wx.EVT_TOOL, self.OnAttach, id=ID.ATTACH)
        f.Bind(wx.EVT_TOOL, self.OnGoBackward, id=wx.ID_BACKWARD)
        f.Bind(wx.EVT_TOOL, self.OnGoForward, id=wx.ID_FORWARD)
        f.Bind(wx.EVT_TOOL, self.OnLoad, id=ID.OPEN)
        f.Bind(wx.EVT_TOOL, self.OnSave, id=wx.ID_SAVE)
        f.Bind(wx.EVT_TOOL, self.OnSaveAs, id=wx.ID_SAVEAS)
        f.Bind(wx.EVT_TOOL, self.OnSetLayout, id=ID.SET_LAYOUT)
        f.threshold_button.Bind(wx.EVT_BUTTON, self.OnThreshold)
        f.grand_av_button.Bind(wx.EVT_BUTTON, self.OnPlotGrandAverage)

        f.Bind(wx.EVT_MENU, self.OnLoad, id=wx.ID_OPEN)
        f.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_CLOSE)
        f.Bind(wx.EVT_MENU, self.OnSave, id=wx.ID_SAVE)
        f.Bind(wx.EVT_MENU, self.OnSaveAs, id=wx.ID_SAVEAS)
        f.Bind(wx.EVT_MENU, self.OnUndo, id=ID.UNDO)
        f.Bind(wx.EVT_MENU, self.OnRedo, id=ID.REDO)
        f.Bind(wx.EVT_MENU, self.OnClear, id=wx.ID_CLEAR)
        f.Bind(wx.EVT_MENU, self.OnSetVLim, id=ID.SET_VLIM)
        f.Bind(wx.EVT_MENU, self.OnSetLayout, id=ID.SET_LAYOUT)
        f.Bind(wx.EVT_MENU, self.OnTogglePlotRange, id=ID.PLOT_RANGE)

        f.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=wx.ID_BACKWARD)
        f.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=wx.ID_FORWARD)
        f.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.REDO)
        f.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=wx.ID_SAVE)
        f.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.UNDO)
        f.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu, id=ID.PLOT_RANGE)

        # canvas events
        f.canvas.mpl_connect('button_press_event', self.OnCanvasClick)
        f.canvas.mpl_connect('key_release_event', self.OnCanvasKey)
#         f.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        self.frame.Show()

    def CanGoBackward(self):
        return self.frame._current_page_i > 0

    def CanGoForward(self):
        return self.frame._current_page_i < self.frame._n_pages - 1

    def CanRedo(self):
        return self.history.can_redo()

    def CanSave(self):
        return bool(self.doc.path)

    def CanUndo(self):
        return self.history.can_undo()

    def CaseChanged(self, index):
        self.frame.CaseChanged(index)

    def GetActiveWindow(self):
        "returns the active window (self, editor, help viewer, ...)"
        if self.frame.IsActive():
            return self.frame
        for c in self.frame.Children:
            if hasattr(c, 'IsActive') and c.IsActive():
                return c
        for w in  wx.GetTopLevelWindows():
            if hasattr(w, 'IsActive') and w.IsActive():
                return w
        return wx.GetActiveWindow()

    def Load(self, path):
        try:
            self.model.load(path)
#             self.frame.ShowPage()
        except Exception as ex:
            msg = str(ex)
            title = "Error Loading Rejections"
            wx.MessageBox(msg, title, wx.ICON_ERROR)
            raise

    def OnAttach(self, event):
        pass

    def OnCanvasClick(self, event):
        "called by mouse clicks"
        log_msg = "Canvas Click:"
        ax = event.inaxes
        if ax:
            log_msg += " ax.ax_idx=%i" % ax.ax_idx
            if ax.ax_idx >= 0:
                idx = ax.epoch_idx
                state = not self.doc.accept[idx]
                tag = "manual"
                desc = "Epoch %i %s" % (idx, state)
                self.model.set_case(idx, state, tag, desc)
            elif ax.ax_idx == -2:
                self.frame.open_topomap()

        logger.debug(log_msg)

    def OnCanvasKey(self, event):
        # GUI Control events
        if event.key == 'right':
            if self.CanGoForward():
                self.OnGoForward(None)
            return
        elif event.key == 'left':
            if self.CanGoBackward():
                self.OnGoBackward(None)
            return
        elif event.key == 'u':
            if self.CanUndo():
                self.OnUndo(None)
            return
        elif event.key == 'U':
            if self.CanRedo():
                self.OnRedo(None)
            return

        # plotting
        ax = event.inaxes
        if ax is None:
            return
        time = event.xdata
        ax_index = getattr(ax, 'ax_idx', None)
        if ax_index == -2:
            return
        if event.key == 't':
            self.frame.PlotTopomap(ax_index, time)
        elif (event.key == 'b'):
            self.frame.PlotButterfly(ax_index)
        elif (event.key == 'c'):
            self.frame.PlotCorrelation(ax_index)

    def OnClear(self, event):
        self.model.clear()

    def OnClose(self, event):
        win = self.GetActiveWindow()
        if win:
            win.Close()
        else:
            event.Skip()

    def OnCloseEvent(self, event):
        "Ask to save unsaved changes"
        if event.CanVeto() and not self.history.is_saved():
            msg = ("The current document has unsaved changes. Would you like "
                   "to save them?")
            cap = ("Saved Unsaved Changes?")
            style = wx.YES | wx.NO | wx.CANCEL | wx.YES_DEFAULT
            cmd = wx.MessageBox(msg, cap, style)
            if cmd == wx.YES:
                if self.Save() != wx.ID_OK:
                    return
            elif cmd == wx.CANCEL:
                return
            elif cmd != wx.NO:
                raise RuntimeError("Unknown answer: %r" % cmd)

        event.Skip()

    def OnGoBackward(self, event):
        "turns the page backward"
        self.ShowPage(self.frame._current_page_i - 1)

    def OnGoForward(self, event):
        "turns the page forward"
        self.ShowPage(self.frame._current_page_i + 1)

#     def OnKeyDown(self, event):
#         logger.debug("OnKeyDown: %s" % event)

    def OnLoad(self, event):
        msg = ("Load the epoch selection from a file.")
        if self.doc.path:
            default_dir, default_name = os.path.split(self.doc.path)
        else:
            default_dir = ''
            default_name = ''
        wildcard = ("Tab Separated Text (*.txt)|*.txt|"
                    "Pickle (*.pickled)|*.pickled")
        dlg = wx.FileDialog(self.frame, msg, default_dir, default_name,
                            wildcard, wx.FD_OPEN)
        rcode = dlg.ShowModal()
        if rcode == wx.ID_OK:
            path = dlg.GetPath()
            self.Load(path)

        dlg.Destroy()
        return rcode

    def OnPlotGrandAverage(self, event):
        self.frame.PlotGrandAverage()

    def OnRedo(self, event):
        self.history.redo()

    def OnSave(self, event):
        self.Save()

    def OnSaveAs(self, event):
        self.SaveAs()

    def OnSetLayout(self, event):
        caption = "Set Plot Layout"
        msg = ("Number of epoch plots for square layout (e.g., '10') or \n"
               "exact n_rows and n_columns (e.g., '5 4'). Add 'nomean' to \n"
               "turn off plotting the page mean at the bottom right (e.g., "
               "'3 nomean').")
        default = ""
        dlg = wx.TextEntryDialog(self.frame, msg, caption, default)
        while True:
            if dlg.ShowModal() == wx.ID_OK:
                nplots = None
                topo = True
                mean = True
                err = []

                # separate options from layout
                value = dlg.GetValue()
                items = value.split(' ')
                options = []
                while not items[-1].isdigit():
                    options.append(items.pop(-1))

                # extract options
                for option in options:
                    if option == 'nomean':
                        mean = False
                    elif option == 'notopo':
                        topo = False
                    else:
                        err.append('Unknown option: "%s"' % option)

                # extract layout info
                if len(items) == 1 and items[0].isdigit():
                    nplots = int(items[0])
                elif len(items) == 2 and all(item.isdigit() for item in items):
                    nplots = tuple(int(item) for item in items)
                else:
                    value_ = ' '.join(items)
                    err = 'Invalid layout specification: "%s"' % value_

                # if all ok: break
                if nplots and not err:
                    break

                # error
                caption = 'Invalid Layout Entry: "%s"' % value
                err.append('Please read the instructions and try again.')
                msg = '\n'.join(err)
                style = wx.OK | wx.ICON_ERROR
                wx.MessageBox(msg, caption, style)
            else:
                dlg.Destroy()
                return

        dlg.Destroy()
        self.frame.SetLayout(nplots, topo, mean)

    def OnSetVLim(self, event):
        default = str(self.frame._vlims.values()[0][1])
        dlg = wx.TextEntryDialog(self.frame, "New Y-axis limit:",
                                 "Set Y-Axis Limit", default)

        if dlg.ShowModal() == wx.ID_OK:
            value = dlg.GetValue()
            try:
                vlim = abs(float(value))
            except Exception as exception:
                msg = wx.MessageDialog(self.frame, str(exception), "Invalid "
                                        "Entry", wx.ICON_ERROR)
                msg.ShowModal()
                msg.Destroy()
                raise
            self.frame.SetVLim(vlim)
        dlg.Destroy()

    def OnThreshold(self, event):
        method = self.config.Read("Threshold/method", "p2p")
        mark_above = self.config.ReadBool("Threshold/mark_above", True)
        mark_below = self.config.ReadBool("Threshold/mark_below", False)
        threshold = self.config.ReadFloat("Threshold/threshold", 2e-12)

        dlg = ThresholdDialog(self.frame, method, mark_above, mark_below,
                              threshold)
        if dlg.ShowModal() == wx.ID_OK:
            threshold = dlg.GetThreshold()
            method = dlg.GetMethod()
            mark_above = dlg.GetMarkAbove()
            if mark_above:
                above = False
            else:
                above = None

            mark_below = dlg.GetMarkBelow()
            if mark_below:
                below = True
            else:
                below = None

            self.model.auto_reject(threshold, method, above, below)

            self.config.Write("Threshold/method", method)
            self.config.WriteBool("Threshold/mark_above", mark_above)
            self.config.WriteBool("Threshold/mark_below", mark_below)
            self.config.WriteFloat("Threshold/threshold", threshold)
            self.config.Flush()

        dlg.Destroy()

    def OnTogglePlotRange(self, event):
        plot_range = event.IsChecked()
        self.frame.SetPlotStyle(plot_range=plot_range)

    def OnUndo(self, event):
        self.history.undo()

    def OnUpdateMenu(self, event):
        id_ = event.GetId()
        if id_ == wx.ID_BACKWARD:
            event.Enable(self.CanGoBackward())
        elif id_ == wx.ID_FORWARD:
            event.Enable(self.CanGoForward())
        elif id_ == ID.UNDO:
            event.Enable(self.CanUndo())
        elif id_ == ID.REDO:
            event.Enable(self.CanRedo())
        elif id_ == wx.ID_SAVE:
            event.Enable(self.CanSave())
        elif id_ == ID.PLOT_RANGE:
            check = self.frame._bfly_kwargs['plot_range']
            event.Check(check)

    def Save(self):
        if self.doc.path:
            self.model.save()
            return wx.ID_OK
        else:
            return self.SaveAs()

    def SaveAs(self):
        msg = ("Save the epoch selection to a file.")
        if self.doc.path:
            default_dir, default_name = os.path.split(self.doc.path)
        else:
            default_dir = ''
            default_name = ''
        wildcard = ("Tab Separated Text (*.txt)|*.txt|"
                    "Pickle (*.pickled)|*.pickled")
        dlg = wx.FileDialog(self.frame, msg, default_dir, default_name,
                            wildcard, wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        rcode = dlg.ShowModal()
        if rcode == wx.ID_OK:
            path = dlg.GetPath()
            self.model.save_as(path)

        dlg.Destroy()
        return rcode

    def ShowPage(self, page):
        self.frame.ShowPage(page)


class TerminalInterface(object):
    """
    Keyboard
    --------
    right-arrow:
        Go to next page.
    left-arrow:
        Go to previous page.
    b:
        Butterfly plot of the currently displayed epoch.
    c:
        Pairwise sensor correlation plot or the current epoch.
    t:
        Topomap plot of the currently displayed time point.
    u:
        Undo.
    shift-u:
        Redo.
    """
    def __init__(self, ds, data='meg', accept='accept', blink='blink',
                 tag='rej_tag', trigger='trigger',
                 path=None, nplots=None, topo=None, mean=None,
                 vlim=None, plot_range=True, color=None, lw=0.2, mark=None,
                 mcolor='r', mlw=0.8, antialiased=True, pos=None, size=None):
        """
        ds : Dataset | mne.Epochs
            The data for which to select trials. If ds is an mne.Epochs object
            the subsequent parameters up to 'trigger' are irrelevant.
        data : str
            Name of the epochs data in ds.
        accept : str
            Name of the boolean Var in ds to accept or reject epochs.
        blink : str
            Name of the eye tracker data in ds.
        tag : str
            Name of the rejection tag (storing the reason for rejecting a
            specific epoch).
        trigger : str
            Name of the int Var containing the event trigger for each epoch
            (used to assert that when loading a rejection file it comes from
            the same data).
        path : None | str
            Path to the desired rejection file. If the file already exists it
            is loaded as starting values. The extension determines the format
            (*.pickled or *.txt).
        nplots : None | int | tuple of 2 int
            Number of epoch plots per page. Can be an ``int`` to produce a
            square layout with that many epochs, or an ``(n_rows, n_columns)``
            tuple. Default (None): use settings form last session.
        topo : None | bool
            Show a topomap plot of the time point under the mouse cursor.
            Default (None): use settings form last session.
        mean : None | bool
            Show a plot of the page mean at the bottom right of the page.
            Default (None): use settings form last session.
        vlim : None | scalar
            Limit of the epoch plots on the y-axis. If None, a value is
            determined automatically to show all data.
        plot_range : bool
            In the epoch plots, plot the range of the data (instead of plotting
            all sensor traces). This makes drawing of pages quicker, especially
            for data with many sensors (default ``True``).
        color : None | matplotlib color
            Color for primary data (default is black).
        lw : scalar
            Linewidth for normal sensor plots.
        mark : None | index for sensor dim
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        mlw : scalar
            Line width for marked sensor plots.
        antialiased : bool
            Perform Antialiasing on epoch plots (associated with a minor speed
            cost).
        pos : None | tuple of 2 int
            Window position on screen. Default (None): use settings form last
            session.
        size : None | tuple of 2 int
            Window size on screen. Default (None): use settings form last
            session.
        """
        bad_chs = None
        self.doc = Document(ds, data, accept, blink, tag, trigger, path,
                            bad_chs)
        self.model = Model(self.doc)
        self.history = self.model.history

        app = wx.GetApp()
        if app is None:
            logger.debug("No WX App found")
            app = wx.App()
            parent = None
            create_menu = True
        elif hasattr(app, 'shell'):
            logger.debug("Eelbrain found")
            parent = app.shell
            create_menu = False
        else:
            logger.debug("WX App found: %s" % app.AppName)
            parent = app.GetTopWindow()
            create_menu = True

        self.controller = Controller(parent, self.model, nplots, topo, mean,
                                     vlim, plot_range, color, lw, mark, mcolor,
                                     mlw, antialiased, pos, size)
        self.frame = self.controller.frame
        if create_menu:
            self.frame._create_menu()
        app.SetTopWindow(self.frame)
        if not app.IsMainLoopRunning():
            logger.info("Entering MainLoop for Epoch Selection GUI")
            app.MainLoop()


class ThresholdDialog(wx.Dialog):

    _methods = (('absolute', 'abs'),
                ('peak-to-peak', 'p2p'))

    def __init__(self, parent, method, mark_above, mark_below, threshold):
        title = "Threshold Criterion Rejection"
        wx.Dialog.__init__(self, parent, wx.ID_ANY, title)
        sizer = wx.BoxSizer(wx.VERTICAL)

        txt = "Mark epochs based on a threshold criterion"
        ctrl = wx.StaticText(self, wx.ID_ANY, txt)
        sizer.Add(ctrl)

        choices = tuple(m[0] for m in self._methods)
        ctrl = wx.RadioBox(self, wx.ID_ANY, "Method", choices=choices)
        selection = [m[1] for m in self._methods].index(method)
        ctrl.SetSelection(selection)
        sizer.Add(ctrl)
        self.method_ctrl = ctrl

        ctrl = wx.CheckBox(self, wx.ID_ANY, "Reject all epochs exceeding the "
                           "threshold")
        ctrl.SetValue(mark_above)
        sizer.Add(ctrl)
        self.mark_above_ctrl = ctrl

        ctrl = wx.CheckBox(self, wx.ID_ANY, "Accept all epochs not exceeding "
                           "the threshold")
        ctrl.SetValue(mark_below)
        sizer.Add(ctrl)
        self.mark_below_ctrl = ctrl

        float_pattern = "^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
        msg = ("Invalid entry for threshold: {value}. Need a floating\n"
               "point number.")
        validator = REValidator(float_pattern, msg, False)
        ctrl = wx.TextCtrl(self, wx.ID_ANY, str(threshold),
                           validator=validator)
        ctrl.SetHelpText("Threshold value (positive scalar)")
        ctrl.SelectAll()
        sizer.Add(ctrl)
        self.threshold_ctrl = ctrl

        # buttons
        button_sizer = wx.StdDialogButtonSizer()

        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        button_sizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)
        button_sizer.AddButton(btn)

        button_sizer.Realize()
        sizer.Add(button_sizer)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def GetMarkAbove(self):
        return self.mark_above_ctrl.IsChecked()

    def GetMarkBelow(self):
        return self.mark_below_ctrl.IsChecked()

    def GetMethod(self):
        index = self.method_ctrl.GetSelection()
        value = self._methods[index][1]
        return value

    def GetThreshold(self):
        text = self.threshold_ctrl.GetValue()
        value = float(text)
        return value