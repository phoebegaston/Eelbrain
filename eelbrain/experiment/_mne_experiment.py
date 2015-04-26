# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from collections import defaultdict
from distutils.version import LooseVersion
import inspect
from itertools import izip
import logging
import os
import re
import shutil
from warnings import warn

import numpy as np

import mne
from mne.baseline import rescale
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)

from .. import _report
from .. import gui
from .. import load
from .. import plot
from .. import save
from .. import table
from .. import testnd
from .. import Dataset, Factor, Var, NDVar, combine
from .._info import BAD_CHANNELS
from .._names import INTERPOLATE_CHANNELS
from .._mne import source_induced_power, dissolve_label, rename_label, \
    morph_source_space
from ..mne_fixes import write_labels_to_annot
from ..mne_fixes import _interpolate_bads_eeg_epochs
from .._data_obj import (align, UTS, DimensionMismatchError,
                         assert_is_legal_dataset_key)
from ..fmtxt import List, Report
from .._resources import predefined_connectivity
from .._utils import subp, ui, keydefaultdict
from .._utils.mne_utils import fix_annot_names, is_fake_mri
from ._experiment import FileTree


__all__ = ['MneExperiment']
logger = logging.getLogger('eelbrain.experiment')
has_mne_09 = LooseVersion(mne.__version__) >= LooseVersion('0.9')


# Parcellations that FreeSurfer generates
FS_PARCS = ('aparc.a2005s', 'aparc.a2009s', 'aparc', 'aparc.DKTatlas')
# Parcellations that come with fsaverage
FSA_PARCS = ('PALS_B12_Brodmann', 'PALS_B12_Lobes', 'PALS_B12_OrbitoFrontal',
             'PALS_B12_Visuotopic')
# Parcellations that Eelbrain makes
EELBRAIN_PARCS = ('lobes', 'lobes-op', 'lobes-ot')


inv_re = re.compile("(free|fixed|loose\.\d+)-"  # orientation constraint
                    "(\d*\.?\d+)-"  # SNR
                    "(MNE|dSPM|sLORETA)"  # method
                    "(?:(\.\d+)-)?"  # depth weighting
                    "(?:-(pick_normal))?")  # pick normal


def _time_str(t):
    "String for representing a time value"
    if t is None:
        return ''
    else:
        return '%i' % round(t * 1000)


def _time_window_str(window, delim='-'):
    "String for representing a time window"
    return delim.join(map(_time_str, window))


class CacheDict(dict):

    def __init__(self, func, key_vars, *args):
        self._func = func
        self._key_vars = key_vars
        self._args = args

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)

        if isinstance(key, basestring):
            out = self._func(*self._args, **{self._key_vars: key})
        else:
            out = self._func(*self._args, **dict(zip(self._key_vars, key)))

        self[key] = out
        return out


temp = {# MEG
        'experiment': '',
        'modality': ('', 'eeg', 'meeg'),
        'reference': ('', 'mastoids'),  # EEG reference
        'equalize_evoked_count': ('', 'eq'),
        # locations
        'meg-sdir': os.path.join('{root}', 'meg'),
        'meg-dir': os.path.join('{meg-sdir}', '{subject}'),
        'raw-dir': '{meg-dir}',

        # raw input files
        'raw-file': os.path.join('{raw-dir}', '{subject}_{experiment}-raw.fif'),
        'trans-file': os.path.join('{raw-dir}', '{mrisubject}-trans.fif'),
        # log-files (eye-tracker etc.)
        'log-dir': os.path.join('{meg-dir}', 'logs'),
        'log-rnd': '{log-dir}/rand_seq.mat',
        'log-data-file': '{log-dir}/data.txt',
        'log-file': '{log-dir}/log.txt',
        'edf-file': os.path.join('{log-dir}', '*.edf'),

        # created input files
        'bads-file': os.path.join('{raw-dir}', '{subject}_{bads-compound}-bad_channels.txt'),
        'rej-dir': os.path.join('{meg-dir}', 'epoch selection'),
        'rej-file': os.path.join('{rej-dir}', '{experiment}_{sns-kind}_{epoch}-{rej}.pickled'),

        # cache
        'cache-dir': os.path.join('{root}', 'eelbrain-cache'),
        # raw
        'raw-cache-dir': os.path.join('{cache-dir}', 'raw'),
        'raw-cache-base': os.path.join('{raw-cache-dir}', '{subject}', '{experiment} {raw-kind}'),
        'cached-raw-file': '{raw-cache-base}-raw.fif',
        'event-file': '{raw-cache-base}-evts.pickled',
        # mne secondary/forward modeling
        'proj-file': '{raw-cache-base}_{proj}-proj.fif',
        'cov-file': '{raw-cache-base}_{cov}-{cov-rej}-{proj}-cov.fif',
        'cov-info-file': '{raw-cache-base}_{cov}-{cov-rej}-{proj}-cov-info.txt',
        'fwd-file': '{raw-cache-base}_{mrisubject}-{src}-fwd.fif',
        # evoked
        'evoked-dir': os.path.join('{cache-dir}', 'evoked'),
        'evoked-file': os.path.join('{evoked-dir}', '{subject}', '{experiment} '
                                    '{sns-kind} {epoch} {model} {evoked-kind}.pickled'),
        # test files
        'test-dir': os.path.join('{cache-dir}', 'test'),
        'data_parc': 'unmasked',
        'test-file': os.path.join('{test-dir}', '{analysis} {group}',
                                  '{epoch} {test} {test_options} {data_parc}.pickled'),

        # MRIs
        'common_brain': 'fsaverage',
        # MRI base files
        'mri-sdir': os.path.join('{root}', 'mri'),
        'mri-dir': os.path.join('{mri-sdir}', '{mrisubject}'),
        'bem-dir': os.path.join('{mri-dir}', 'bem'),
        'mri-cfg-file': os.path.join('{mri-dir}', 'MRI scaling parameters.cfg'),
        'mri-file': os.path.join('{mri-dir}', 'mri', 'orig.mgz'),
        'bem-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem.fif'),
        'bem-sol-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem-sol.fif'),
        'head-bem-file': os.path.join('{bem-dir}', '{mrisubject}-head.fif'),
        'src-file': os.path.join('{bem-dir}', '{mrisubject}-{src}-src.fif'),
        # Labels
        'hemi': ('lh', 'rh'),
        'label-dir': os.path.join('{mri-dir}', 'label'),
        'annot-file': os.path.join('{label-dir}', '{hemi}.{parc}.annot'),
        # pickled list of labels
        'mri-cache-dir': os.path.join('{cache-dir}', 'mri', '{mrisubject}'),
        'label-file': os.path.join('{mri-cache-dir}', '{parc}.pickled'),

        # (method) plots
        'plot-dir': os.path.join('{root}', 'plots'),
        'plot-file': os.path.join('{plot-dir}', '{analysis}', '{name}.{ext}'),

        # general analysis parameters
        'analysis': '',  # analysis parameters (sns-kind, src-kind, ...)
        'test_options': '',
        'name': '',

        # result output files
        # data processing parameters
        #    > group
        #        > kind of test
        #    > single-subject
        #        > kind of test
        #            > subject
        'folder': '',  # intermediate folder for deep files
        'resname': '',  # analysis name (GA-movie, max plot, ...)
        'ext': 'pickled',  # file extension
        'res-dir': os.path.join('{root}', 'results'),
        'res-file': os.path.join('{res-dir}', '{analysis}', '{resname}.{ext}'),
        'res-deep-file': os.path.join('{res-dir}', '{analysis}', '{folder}',
                                      '{resname}.{ext}'),
        'res-g-file': os.path.join('{res-dir}', '{analysis} {group}',
                                   '{resname}.{ext}'),
        'res-g-deep-file': os.path.join('{res-dir}', '{analysis} {group}',
                                        '{folder}', '{resname}.{ext}'),
        'res-s-file': os.path.join('{res-dir}', '{analysis} subjects',
                                    '{resname}', '{subject}.{ext}'),

        # besa
        'besa-root': os.path.join('{root}', 'besa'),
        'besa-trig': os.path.join('{besa-root}', '{subject}', '{subject}_'
                                  '{experiment}_{epoch}_triggers.txt'),
        'besa-evt': os.path.join('{besa-root}', '{subject}', '{subject}_'
                                 '{experiment}_{epoch}[{rej}].evt'),

        # MRAT
        'mrat_condition': '',
        'mrat-root': os.path.join('{root}', 'mrat'),
        'mrat-sns-root': os.path.join('{mrat-root}', '{sns-kind}',
                                      '{epoch} {model} {evoked-kind}'),
        'mrat-src-root': os.path.join('{mrat-root}', '{src-kind}',
                                      '{epoch} {model} {evoked-kind}'),
        'mrat-sns-file': os.path.join('{mrat-sns-root}', '{mrat_condition}',
                                      '{mrat_condition}_{subject}-ave.fif'),
        'mrat_info-file': os.path.join('{mrat-root}', '{subject} info.txt'),
        'mrat-src-file': os.path.join('{mrat-src-root}', '{mrat_condition}',
                                      '{mrat_condition}_{subject}'),
         }


class MneExperiment(FileTree):
    """Analyze an MEG experiment (gradiometer only) with MNE

    Parameters
    ----------
    root : str | None
        the root directory for the experiment (usually the directory
        containing the 'meg' and 'mri' directories)
    find_subjects : bool
        Automatically look for subjects in the MEG-directory (default
        True). Set ``find_subjects=False`` to initialize the experiment
        without any files.

    Notes
    -----
    .. seealso::
        Guide on using :ref:`experiment-class-guide`.
    """
    path_version = None

    # Experiment Constants
    # ====================

    # add this value to all trigger times
    trigger_shift = 0

    # variables for automatic labeling {name: {trigger: label, triggers: label}}
    variables = {}

    # Default values for epoch definitions
    _epoch_default = {'tmin':-0.1, 'tmax': 0.6, 'decim': 5, 'baseline': (None, 0)}
    epoch_default = {}

    # named epochs
    epochs = {'epoch': dict(sel="stim=='target'"),
              'cov': dict(sel_epoch='epoch', tmin=-0.1, tmax=0)}
    # Rejection
    # =========
    # eog_sns: The sensors to plot separately in the rejection GUI. The default
    # is the two MEG sensors closest to the eyes.
    _eog_sns = {'KIT-NY': ['MEG 143', 'MEG 151'],
                'KIT-AD': ['MEG 087', 'MEG 130'],
                'KIT-BRAINVISION': ['HEOGL', 'HEOGR', 'VEOGb']}
    #
    # epoch_rejection dict:
    #
    # kind : 'auto', 'manual', 'make'
    #     How the rejection is derived; 'auto': use the parameters to do the
    #     selection on the fly; 'manual': manually create a rejection file (use
    #     the selection GUI .make_rej()); 'make' a rejection file
    #     is created by the user
    # cov-rej : str
    #     rej setting to use for cov under this setting.
    #
    # For manual rejection
    # ^^^^^^^^^^^^^^^^^^^^
    # decim : int
    #     Decim factor for the rejection GUI (default is to use epoch setting).
    #
    # For automatic rejection
    # ^^^^^^^^^^^^^^^^^^^^^^^
    # threshod : None | dict
    #     the reject argument when loading epochs:
    # edf : list of str
    #     How to use eye tracker information in rejection. True
    #     causes edf files to be loaded but not used
    #     automatically.
    _epoch_rejection = {'': {'kind': None},
                        'man': {'kind': 'manual',
                                'interpolation': True},
                        'et': {'kind': 'auto',
                               'threshold': dict(mag=3e-12),
                               'edf': ['EBLINK'],
                               },
                        'threshold': {'kind': 'auto',
                                      'threshold': dict(mag=3e-12)}
                        }
    epoch_rejection = {}

    exclude = {}  # field_values to exclude (e.g. subjects)

    # groups can be defined as subject lists: {'group': ('member1', 'member2', ...)}
    # or by exclusion: {'group': {'base': 'all', 'exclude': ('member1', 'member2')}}
    groups = {}

    # whether to look for and load eye tracker data when loading raw files
    has_edf = defaultdict(lambda: False)

    # raw processing settings {name: (args, kwargs)}
    _raw = {'clm': None,
            '0-40': ((None, 40), {'method': 'iir'}),
            '0.16-40': ((0.16, 40), {'l_trans_bandwidth': 0.08,
                                     'filter_length': '60s'}),
            '1-40': ((1, 40), {'method': 'iir'})}

    # projection definition:
    # "base": 'raw' for raw file, or epoch name
    # "rej": rejection setting to use (only applies for epoch projs)
    # r.g. {'ironcross': {'base': 'adj', 'rej': 'man'}}
    projs = {}

    # Pattern for subject names. The first group is used to determine what
    # MEG-system the data was recorded from
    _subject_re = '(R|A|Y|AD|QP)(\d{3,})$'
    _meg_systems = {'R': 'KIT-NY',
                    'A': 'KIT-AD', 'Y': 'KIT-AD', 'AD': 'KIT-AD', 'QP': 'KIT-AD'}

    # kwargs for regularization of the covariance matrix (see .make_cov())
    _covs = {'auto': {'method': 'auto'},
             'bestreg': {'reg': 'best'},
             'reg': {'reg': True},
             'noreg': {'reg': None}}

    # MRI subject names: {subject: mrisubject} mappings
    # selected with e.set(mri=dict_name)
    # default is identity (mrisubject = subject)
    _mri_subjects = {'': keydefaultdict(lambda s: s)}

    # state variables that are always shown in self.__repr__():
    _repr_kwargs = ('subject', 'rej')

    # Where to search for subjects (defined as a template name). If the
    # experiment searches for subjects automatically, it scans this directory
    # for subfolders matching _subject_re.
    _subject_loc = 'meg-sdir'

    # Custom parcellations. Set to None to disable checking on set().
    parcs = ()

    # basic templates to use. Can be a string referring to a templates
    # dictionary in the module level _temp dictionary, or a templates
    # dictionary
    _templates = temp
    # specify additional templates
    _values = {}
    # specify defaults for specific fields (e.g. specify the initial subject
    # name)
    defaults = {}

    # model order: list of factors in the order in which models should be built
    # (default for factors not in this list is alphabetic)
    _model_order = []

    # Backup
    # ------
    # basic state for a backup
    _backup_state = {'subject': '*', 'mrisubject': '*', 'experiment': '*',
                     'raw': 'clm', 'modality': '*'}
    # files to back up, together with state modifications on the basic state
    _backup_files = (('raw-file', {}),
                     ('bads-file', {}),
                     ('rej-file', {'raw': '*', 'epoch': '*', 'rej': '*'}),
                     ('trans-file', {}),
                     ('mri-cfg-file', {}),
                     ('log-dir', {}),)

    # Tests
    # -----
    # specify tests as (test_type, model, test_parameter) tuple. For example,
    # ("anova", "condition", "condition*subject")
    # ("t_contrast_rel", "ref%loc", "+min(ref|left>nref|*, ref|right>nref|*)")
    # Make sure dictionary keys (test names) are appropriate for filenames.
    # tests imply a model which is set automatically
    tests = {}
    cluster_criteria = {'mintime': 0.025, 'minsensor': 4, 'minsource': 10}

    def __init__(self, root=None, find_subjects=True, **state):
        # create attributes (overwrite class attributes)
        self._subject_re = re.compile(self._subject_re)
        self.groups = self.groups.copy()
        self.projs = self.projs.copy()
        self.cluster_criteria = self.cluster_criteria.copy()
        self._mri_subjects = self._mri_subjects.copy()
        self._templates = self._templates.copy()
        # templates version
        if self.path_version is None or self.path_version == 0:
            self._templates['raw-dir'] = os.path.join('{meg-dir}', 'raw')
            self._templates['raw-file'] = os.path.join('{raw-dir}', '{subject}_'
                                              '{experiment}_{raw-kind}-raw.fif')
        elif self.path_version != 1:
            raise ValueError("path_version needs to be 0 or 1")
        # update templates with _values
        for cls in reversed(inspect.getmro(self.__class__)):
            if hasattr(cls, '_values'):
                self._templates.update(cls._values)

        ########################################################################
        # variables
        self.variables = self.variables.copy()
        for k, v in self.variables.iteritems():
            assert_is_legal_dataset_key(k)
            triggers = []
            for trigger, name in v.iteritems():
                if not isinstance(name, basestring):
                    raise TypeError("Invalid cell name in variable "
                                    "definition: %s" % repr(name))

                if isinstance(trigger, tuple):
                    triggers.extend(trigger)
                else:
                    triggers.append(trigger)

            if not all(isinstance(t, int) or t == 'default' for t in triggers):
                raise TypeError("Invalid trigger code in variable "
                                "definition : %s" % repr(trigger))

        ########################################################################
        # epochs
        epochs = {}
        super_epochs = {}
        for name, parameters in self.epochs.iteritems():
            # filter out super_epochs
            if 'sub_epochs' in parameters:
                if len(parameters) > 1:
                    msg = ("Super-epochs can only have one parameters called "
                           "'sub_epochs'; got %r" % parameters)
                    raise ValueError(msg)
                epoch = parameters.copy()
                epoch['name'] = name
                super_epochs[name] = epoch

            # expand epoch dict
            epoch = self._epoch_default.copy()
            epoch.update(self.epoch_default)
            epoch.update(self.epochs[name])
            epoch['name'] = name

            epochs[name] = epoch
        # re-integrate super-epochs
        for name, parameters in super_epochs.iteritems():
            sub_epochs = self.epochs[name]['sub_epochs']

            # make sure definition is not recursive
            if any('sub_epochs' in epochs[n] for n in sub_epochs):
                msg = ("Super-epochs can't be defined recursively (%r)" % name)
                raise ValueError(msg)

            # find params
            for param in ('tmin', 'tmax', 'decim'):
                values = set(epochs[n][param] for n in sub_epochs)
                if len(values) > 1:
                    param_repr = ', '.join(repr(v) for v in values)
                    msg = ("All sub_epochs of a super-epoch must have the "
                           "same setting for %r; %r got {%s}."
                           % (param, name, param_repr))
                    raise ValueError(msg)
                parameters[param] = values.pop()
        epochs.update(super_epochs)
        # find relevant rej-files (for cache checking)
        def _rej_epochs(epoch):
            "Find which rej-files an epoch depends on"
            if 'sub_epochs' in epoch:
                names = epoch['sub_epochs']
                return sum((_rej_epochs(epochs[n]) for n in names), ())
            elif 'sel_epoch' in epoch:
                return _rej_epochs(epochs[epoch['sel_epoch']])
            else:
                return (epoch['name'],)
        for name, epoch in epochs.iteritems():
            if 'sub_epochs' in epoch or 'sel_epoch' in epoch:
                epoch['_rej_file_epochs'] = _rej_epochs(epoch)
        self.epochs = epochs

        ########################################################################
        # store epoch rejection settings
        epoch_rejection = self._epoch_rejection.copy()
        epoch_rejection.update(self.epoch_rejection)
        self.epoch_rejection = epoch_rejection

        ########################################################################
        # tests
        tests = {}
        for test, params in self.tests.iteritems():
            # backwards compatibility for old test specification
            if isinstance(params, (tuple, list)):
                warn("MneExperiment.tests should be defined as dictionaries, "
                     "test definitions with tuples/lists is deprecated. Please "
                     "change your MneExperiment subclass definition.",
                     DeprecationWarning)
                kind, model, test_parameter = params
                if kind == 'anova':
                    params = {'kind': kind, 'model': model, 'x': test_parameter}
                elif kind == 'ttest_rel':
                    m = re.match(r"\s*([\w|]+)\s*([<=>])\s*([\w|]+)$", test_parameter)
                    if m is None:
                        raise ValueError("The contrast definition %s for test "
                                         "%s could not be parsed." %
                                         (repr(test_parameter), test))
                    c1, tail, c0 = m.groups()
                    if '|' in c1:
                        c1 = tuple(c1.split('|'))
                        c0 = tuple(c0.split('|'))

                    if tail == '=':
                        tail = 0
                    elif tail == '>':
                        tail = 1
                    elif tail == '<':
                        tail = -1
                    else:
                        raise ValueError("%r in t-test contrast=%r"
                                         % (tail, test_parameter))
                    params = {'kind': kind, 'model': model, 'c1': c1, 'c0': c0,
                              'tail': tail}
                elif kind == 't_contrast_rel':
                    params = {'kind': kind, 'model': model,
                              'contrast': test_parameter}
                else:
                    raise ValueError("Unknown test: %s" % repr(kind))
            elif not isinstance(params, dict):
                raise TypeError("Tests need to be specified as dictionary, "
                                "got %s" % repr(params))

            # test descriptions
            kind = params['kind']
            if kind == 'anova':
                desc = params['x']
            elif kind == 'ttest_rel':
                tail = params.get('tail', 0)
                if tail == 0:
                    link = ' = '
                elif tail > 0:
                    link = ' > '
                else:
                    link = ' < '
                desc = link.join((params['c1'], params['c0']))
            elif kind == 't_contrast_rel':
                desc = params['contrast']
            elif kind == 'custom':
                desc = 'Custom test'
            else:
                raise NotImplementedError("Invalid test kind: %r" % kind)
            params['desc'] = desc

            tests[test] = params
        self._tests = tests

        ########################################################################
        # Experiment class setup
        FileTree.__init__(self)

        # register variables with complex behavior
        self._register_field('raw', sorted(self._raw))
        self._register_field('rej', self.epoch_rejection.keys(), 'man',
                             post_set_handler=self._post_set_rej)
        self._register_field('group', self.groups.keys() + ['all'], 'all',
                             eval_handler=self._eval_group,
                             post_set_handler=self._post_set_group)
        # epoch
        epoch_keys = sorted(self.epochs)
        for default_epoch in epoch_keys:
            if 'sel_epoch' not in self.epochs[default_epoch]:
                break
        else:
            default_epoch = None
        self._register_field('epoch', epoch_keys, default_epoch)
        # cov
        if 'bestreg' in self._covs:
            default_cov = 'bestreg'
        else:
            default_cov = None
        self._register_field('cov', sorted(self._covs), default_cov)
        self._register_field('mri', sorted(self._mri_subjects))
        self._register_field('inv', default='free-3-dSPM',
                             eval_handler=self._eval_inv,
                             post_set_handler=self._post_set_inv)
        self._register_field('model', eval_handler=self._eval_model)
        self._register_field('test', sorted(self._tests) or None,
                             post_set_handler=self._post_set_test)
        self._register_field('parc', default='aparc',
                             eval_handler=self._eval_parc)
        self._register_field('proj', [''] + self.projs.keys())
        self._register_field('src', ('ico-4', 'vol-10', 'vol-7', 'vol-5'))
        self._register_field('mrisubject')
        self._register_field('subject')

        # compounds
        self._register_compound('bads-compound', ('experiment', 'modality'))
        self._register_compound('raw-kind', ('modality', 'raw'))
        self._register_compound('sns-kind', ('raw-kind', 'proj'))
        self._register_compound('src-kind', ('sns-kind', 'cov', 'mri', 'inv'))
        self._register_compound('evoked-kind', ('rej', 'equalize_evoked_count'))
        self._register_compound('eeg-kind', ('sns-kind', 'reference'))

        # Define make handlers
        self._bind_cache('cached-raw-file', self.make_raw)
        self._bind_cache('cov-file', self.make_cov)
        self._bind_cache('src-file', self.make_src)
        self._bind_cache('fwd-file', self.make_fwd)
        self._bind_make('label-file', self.make_labels)

        # Check that the template model is complete
        self._find_missing_fields()

        # set initial values
        self.set(**state)
        self.set_root(root, find_subjects)
        self._post_set_group(None, self.get('group'))
        self.store_state()
        self.brain = None

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        for subject in self.iter():
            yield subject

    def _annot_mtime(self):
        "Return max mtime of annot files or None if they do not exist."
        mtime = 0
        for _ in self.iter('hemi'):
            fpath = self.get('annot-file')
            if os.path.exists(fpath):
                mtime = max(mtime, os.path.getmtime(fpath))
            else:
                return
        return mtime

    def _process_subject_arg(self, subject, kwargs):
        """Process subject arg for methods that work on groups and subjects

        Returns
        -------
        subject : None | str
            Subject name if the value specifies a subject, None otherwise.
        group : None | str
            Group name if the value specifies a group, None otherwise.
        """
        if subject is None:
            group = None
            subject_ = self.get('subject', **kwargs)
        elif subject is True:
            group = self.get('group', **kwargs)
            subject_ = None
        elif subject in self.get_field_values('group'):
            group = subject
            subject_ = None
            self.set(group=group, **kwargs)
        else:
            group = None
            subject_ = subject
            self.set(subject=subject, **kwargs)

        return subject_, group

    def add_epochs_stc(self, ds, ndvar=True, baseline=None, morph=False):
        """
        Transform epochs contained in ds into source space (adds a list of mne
        SourceEstimates to ds)

        Parameters
        ----------
        ds : Dataset
            The Dataset containing the mne Epochs for the desired trials.
        ndvar : bool
            Add the source estimates as NDVar named 'src' (default). Set to
            False to add a list of MNE SourceEstimate objects named 'stc'.
        baseline : None | True | tuple
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to apply no baseline
            correction (None).
        morph : bool
            Morph the source estimates to the common_brain (default False).
        """
        subject = ds['subject']
        if len(subject.cells) != 1:
            err = "ds must have a subject variable with exactly one subject"
            raise ValueError(err)
        subject = subject.cells[0]
        self.set(subject=subject)
        if baseline is True:
            baseline = self.epochs[self.get('epoch')]['baseline']

        epochs = ds['epochs']
        inv = self.load_inv(epochs)
        stc = apply_inverse_epochs(epochs, inv, **self._params['apply_inv_kw'])

        if ndvar:
            self.make_annot()
            subject = self.get('mrisubject')
            src = self.get('src')
            mri_sdir = self.get('mri-sdir')
            parc = self.get('parc') or None
            src = load.fiff.stc_ndvar(stc, subject, src, mri_sdir,
                                      self._params['apply_inv_kw']['method'],
                                      self._params['make_inv_kw'].get('fixed', False),
                                      parc=parc)
            if baseline is not None:
                src -= src.summary(time=baseline)

            if morph:
                ds['srcm'] = morph_source_space(src, self.get('common_brain'))
            else:
                ds['src'] = src
        else:
            if baseline is not None:
                raise NotImplementedError("Baseline for SourceEstimate")
            if morph:
                raise NotImplementedError("Morphing for SourceEstimate")
            ds['stc'] = stc

    def add_evoked_stc(self, ds, ind_stc=False, ind_ndvar=False, morph_stc=False,
                       morph_ndvar=False, baseline=None, keep_evoked=False):
        """
        Add source estimates to a dataset with evoked data.

        Parameters
        ----------
        ds : Dataset
            The Dataset containing the Evoked objects.
        ind_stc : bool
            Add source estimates on individual brains as list of
            :class:`mne.SourceEstimate`.
        ind_ndvar : bool
            Add source estimates on individual brain as :class:`NDVar` (only
            possible for datasets containing data of a single subject).
        morph_stc : bool
            Add source estimates morphed to the common brain as list of
            :class:`mne.SourceEstimate`.
        morph_ndvar : bool
            Add source estimates morphed to the common brain as :class:`NDVar`.
        baseline : None | True | tuple
            Apply baseline correction (in source space) using this period. True
            to use the epoch's baseline specification. The default is to not
            apply baseline correction (None).
        keep_evoked : bool
            Keep the sensor space data in the Dataset that is returned (default
            False).

        Notes
        -----
        Assumes that all Evoked of the same subject share the same inverse
        operator.
        """
        if not any((ind_stc, ind_ndvar, morph_stc, morph_ndvar)):
            err = ("Nothing to load, set at least one of (ind_stc, ind_ndvar, "
                   "morph_stc, morph_ndvar) to True")
            raise ValueError(err)

        if isinstance(baseline, str):
            raise NotImplementedError("Baseline form different epoch")
        elif baseline is True:
            baseline = self.epochs[self.get('epoch')]['baseline']

        # find from subjects
        common_brain = self.get('common_brain')
        meg_subjects = ds.eval('subject.cells')
        n_subjects = len(meg_subjects)
        if ind_ndvar and n_subjects > 1:
            err = ("Can't use ind_ndvar with data from more than one "
                   "subjects; an NDVar can only be created from stcs that are "
                   "estimated on the same brain. Use morph_ndvar=True "
                   "instead.")
            raise ValueError(err)
        from_subjects = {}  # from-subject for the purpose of morphing
        for subject in meg_subjects:
            if is_fake_mri(self.get('mri-dir', subject=subject)):
                subject_from = common_brain
            else:
                subject_from = self.get('mrisubject', subject=subject)
            from_subjects[subject] = subject_from

        collect_morphed_stcs = (morph_stc or morph_ndvar)
        collect_ind_stcs = ind_stc or ind_ndvar

        # make sure annot files are available (needed only for NDVar)
        all_are_common_brain = all(v == common_brain for v in from_subjects.values())
        if (ind_ndvar and all_are_common_brain) or morph_ndvar:
            self.make_annot(mrisubject=common_brain)
        if ind_ndvar and not all_are_common_brain:
            self.make_annot(mrisubject=from_subjects[meg_subjects[0]])

        # convert evoked objects
        stcs = []
        mstcs = []
        invs = {}
        mm_cache = CacheDict(self.load_morph_matrix, 'mrisubject')
        for subject, evoked in izip(ds['subject'], ds['evoked']):
            subject_from = from_subjects[subject]

            # get inv
            if subject in invs:
                inv = invs[subject]
            else:
                inv = invs[subject] = self.load_inv(evoked, subject=subject)

            # apply inv
            stc = apply_inverse(evoked, inv, **self._params['apply_inv_kw'])

            # baseline correction
            if baseline:
                rescale(stc._data, stc.times, baseline, 'mean', copy=False)

            if collect_ind_stcs:
                stcs.append(stc)

            if collect_morphed_stcs:
                if subject_from == common_brain:
                    if ind_stc:
                        stc = stc.copy()
                    stc.subject = common_brain
                else:
                    mm, v_to = mm_cache[subject_from]
                    stc = mne.morph_data_precomputed(subject_from, subject,
                                                     stc, v_to, mm)
                mstcs.append(stc)

        # add to Dataset
        src = self.get('src')
        parc = self.get('parc') or None
        mri_sdir = self.get('mri-sdir')
        # for name, key in izip(do, keys):
        if ind_stc:
            ds['stc'] = stcs
        if ind_ndvar:
            subject = from_subjects[meg_subjects[0]]
            ds['src'] = load.fiff.stc_ndvar(stcs, subject, src, mri_sdir,
                                            self._params['apply_inv_kw']['method'],
                                            self._params['make_inv_kw'].get('fixed', False),
                                            parc=parc)
        if morph_stc or morph_ndvar:
            if morph_stc:
                ds['stcm'] = mstcs
            if morph_ndvar:
                ds['srcm'] = load.fiff.stc_ndvar(mstcs, common_brain, src, mri_sdir,
                                                 self._params['apply_inv_kw']['method'],
                                                 self._params['make_inv_kw'].get('fixed', False),
                                                 parc=parc)

        if not keep_evoked:
            del ds['evoked']

    @staticmethod
    def _add_stc_label(ds, label, label_cache):
        """
        Extract the label time course from a list of SourceEstimates.

        Returns nothing.

        Parameters
        ----------
        ds : Dataset
            Dataset containing a list of SourceEstimates and a subject
            variabls.
        label :
            the label's name (e.g., 'fusiform_lh').
        labels : CacheDict
            Labels cache.

        Returns
        -------
        key : str
            The key under which the label was added to the Dataset.
        """
        label_id = 0
        if 'label_ids' not in ds.info:
            key = 'label_tc_0'
            ds.info['label_ids'] = {}
        elif label in ds.info['label_ids']:
            raise RuntimeError("Label already added: %r" % label)
        else:
            while 'label_tc_%i' % label_id in ds.info['label_ids']:
                label_id += 1
            key = 'label_tc_%i' % label_id
        ds.info['label_ids'][label] = key

        x = []
        for case in ds.itercases():
            stc = case['stc']
            label_data = stc.in_label(label_cache[case['subject']][label]).data
            x.append(label_data.mean(0))

        time = UTS(stc.tmin, stc.tstep, stc.shape[1])
        ds[key] = NDVar(np.array(x), dims=('case', time))
        return key

    def backup(self, dst_root):
        """Backup all essential files to ``dst_root``.

        Parameters
        ----------
        dst_root : str
            Directory to use as root for the backup.

        Notes
        -----
        For repeated backups ``dst_root`` can be the same. If a file has been
        previously backed up, it is only copied if the local copy has been
        modified more recently than the previous backup. If the backup has been
        modified more recently than the local copy, a warning is displayed.

        Currently, the following files are included in the backup::

         * Calmed raw file (raw='clm')
         * Bad channels file
         * All rejection files
         * The trans-file
         * All files in the ``meg/{subject}/logs`` directory
         * For scaled MRIs, the file specifying the scale parameters

        MRIs are currently not backed up.
        """
        root = self.get('root')
        root_len = len(root) + 1

        dirs = []  # directories to create
        pairs = []  # (src, dst) pairs to copy
        for temp, state_mod in self._backup_files:
            # determine state
            if state_mod:
                state = self._backup_state.copy()
                state.update(state_mod)
            else:
                state = self._backup_state

            # find files to back up
            if temp.endswith('dir'):
                paths = []
                for dirpath in self.glob(temp, **state):
                    for root_, _, filenames in os.walk(dirpath):
                        paths.extend(os.path.join(root_, fn) for fn in filenames)
            else:
                paths = self.glob(temp, **state)

            # convert to (src, dst) pairs
            for src in paths:
                if not src.startswith(root):
                    raise ValueError("Can only backup files in root directory")
                tail = src[root_len:]
                dst = os.path.join(dst_root, tail)
                if os.path.exists(dst):
                    src_m = os.path.getmtime(src)
                    dst_m = os.path.getmtime(dst)
                    if dst_m == src_m:
                        continue
                    elif dst_m > src_m:
                        msg = "Backup more recent than original: %s" % tail
                        logger.warn(msg)
                        continue
                else:
                    i = 0
                    while True:
                        i = tail.find(os.sep, i + 1)
                        if i == -1:
                            break
                        path = tail[:i]
                        if path not in dirs:
                            dirs.append(path)

                pairs.append((src, dst))

        if len(pairs) == 0:
            logger.info("All files backed up.")
            return

        logger.info("Backing up %i files ..." % len(pairs))
        # create directories
        for dirname in dirs:
            dirpath = os.path.join(dst_root, dirname)
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)
        # copy files
        for src, dst in pairs:
            shutil.copy2(src, dst)

    def clear_cache(self, level=1):
        """Remove cached files.

        Parameters
        ----------
        level : int
            Level up to which to clear the cache (see notes below). The default
            is 1, which deletes all cached files.

        Notes
        -----
        Each lower level subsumes the higher levels:

        ``1``
            data files - these need to be cleared when anything about the
            epoch definition changes (tmin, tmax, event inclusion, ...). Note
            that you might also have to manually update epoch rejection files
            with the :meth:`MneExperiment.make_rej` method.
        ``5``
            tests - these need to be cleared when the members of the relevant
            subject groups change.

        Examples
        --------
        To delete only test files, after adding raw data for a new subject to
        the experiment::

            >>> e.clear_cache(1)

        To delete cached data files, after changing the selection criteria for
        a secondary epoch::

            >>> e.clear_cache(5)

        If criteria on a primary epoch are changed, the trial rejection has to
        be re-done in addition to clearing the cache.
        """
        if level <= 1:
            self.rm('cache-dir', confirm=True)
            print "Cached data cleared."
        else:
            self.rm('test-dir', confirm=True)
            print "Cached tests cleared."

    def _fix_eeg_ndvar(self, ndvar, apply_standard_montag):
        # connectivity
        ndvar.sensor.set_connectivity(predefined_connectivity('BrainCap32Ch'))
        # montage
        if apply_standard_montag:
            m = mne.channels.read_montage('easycap-M1')
            m.ch_names = [n.upper() for n in m.ch_names]
            m.ch_names[m.ch_names.index('TP9')] = 'A1'
            m.ch_names[m.ch_names.index('TP10')] = 'A2'
            m.ch_names[m.ch_names.index('FP2')] = 'VEOGt'
            ndvar.sensor.set_sensor_positions(m)
        # reference
        reference = self.get('reference')
        if reference:
            if reference == 'mastoids':
                ndvar -= ndvar.summary(sensor=['A1', 'A2'])
            else:
                raise ValueError("Unknown reference: reference=%r" % reference)

    def get_field_values(self, field, exclude=True):
        """Find values for a field taking into account exclusion

        Parameters
        ----------
        field : str
            Field for which to find values.
        exclude : bool | list of values
            Exclude values. If True, exclude values based on ``self.exclude``.
            For 'mrisubject', exclusions are done on 'subject'. For 'group',
            no exclusions are done.
        """
        if exclude is True:
            exclude = self.exclude.get(field, None)
        elif isinstance(exclude, basestring):
            exclude = (exclude,)

        if field == 'mrisubject':
            subjects = FileTree.get_field_values(self, 'subject')
            mri_subjects = self._mri_subjects[self.get('mri')]
            mrisubjects = sorted(mri_subjects[s] for s in subjects)
            if exclude:
                mrisubjects = [s for s in mrisubjects if s not in exclude]
            common_brain = self.get('common_brain')
            if common_brain and (not exclude or common_brain not in exclude):
                mrisubjects.insert(0, common_brain)
            return mrisubjects
        elif field == 'group':
            values = ['all', 'all!']
            values.extend(self.groups.keys())
            if exclude:
                values = [v for v in values if v not in exclude]
            return values
        else:
            return FileTree.get_field_values(self, field, exclude)

    def _get_group_members(self, group):
        "For groups except all and all!"
        if group == 'all':
            return self.get_field_values('subject')
        elif group == 'all!':
            return self.get_field_values('subject', False)

        group_def = self.groups[group]
        if isinstance(group_def, dict):
            base = self._get_group_members(group_def.get('base', 'all'))
            exclude = group_def['exclude']
            if isinstance(exclude, basestring):
                exclude = (exclude,)
            elif not isinstance(exclude, (tuple, list, set)):
                msg = ("exclusion must be defined as str | tuple | list | set; got "
                       "%s" % repr(exclude))
                raise TypeError(msg)
            return [s for s in base if s not in exclude]
        elif isinstance(group_def, (list, tuple)):
            return [s for s in self._get_group_members('all') if s in group_def]
        else:
            raise TypeError("group %s=%r" % (group, group_def))

    def _get_raw_path(self, make=False):
        if self._raw[self.get('raw')] is None:
            return self.get('raw-file')
        else:
            return self.get('cached-raw-file', make=make)

    def iter(self, fields='subject', exclude=True, values={}, group=None, *args,
             **kwargs):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : list | str
            Field(s) over which should be iterated.
        exclude : dict  {str: str, str: iterator over str, ...}
            Values to exclude from the iteration with {name: value} and/or
            {name: (sequence of values, )} entries.
        values : dict  {str: iterator over str}
            Fields with custom values to iterate over (instead of the
            corresponding field values) with {name: (sequence of values)}
            entries.
        group : None | str
            If iterating over subjects, use this group ('all' for all except
            excluded subjects, 'all!' for all including excluded subjects, or
            a name defined in experiment.groups).
        mail : bool | str
            Send an email when iteration is finished. Can be True or an email
            address. If True, the notification is sent to :attr:`.owner`.
        prog : bool | str
            Show a progress dialog; str for dialog title.
        *others* :
            Fields with constant values throughout the iteration.
        """
        if group is not None:
            self.set(group=group)

        if 'subject' in fields and 'subject' not in values:
            if group is None:
                group = self.get('group')
            values = values.copy()
            values['subject'] = self._get_group_members(group)

        return FileTree.iter(self, fields, exclude, values, *args, **kwargs)

    def iter_range(self, start=None, stop=None, field='subject'):
        """Iterate through a range on a field with ordered values.

        Parameters
        ----------
        start : None | str
            Start value (inclusive). With ``None``, begin at the first value.
        stop : None | str
            Stop value (inclusive). With ``None``, end with the last value.
        field : str
            Name of the field.

        Returns
        -------
        iterator over value : str
            Current field value.
        """
        values = self.get_field_values(field)
        if start is not None:
            start = values.index(start)
        if stop is not None:
            stop = values.index(stop) + 1
        idx = slice(start, stop)
        values = values[idx]

        with self._temporary_state:
            for value in values:
                self.restore_state(discard_tip=False)
                self.set(**{field: value})
                yield value

    def label_events(self, ds, experiment=None, subject=None):
        """
        Adds T (time) and SOA (stimulus onset asynchrony) to the Dataset.

        Parameters
        ----------
        ds : Dataset
            A Dataset containing events (as returned by
            :func:`load.fiff.events`).

        Notes
        -----
        Subclass this method to specify events.
        """
        if self.trigger_shift:
            ds['i_start'] += round(self.trigger_shift * ds.info['raw'].info['sfreq'])

        if 'raw' in ds.info:
            raw = ds.info['raw']
            sfreq = raw.info['sfreq']
            ds['T'] = ds['i_start'] / sfreq
            ds['SOA'] = Var(np.ediff1d(ds['T'].x, 0))

        for name, coding in self.variables.iteritems():
            ds[name] = ds['trigger'].as_factor(coding, name)

        # add subject label
        ds['subject'] = Factor([ds.info['subject']], repeat=ds.n_cases, random=True)
        return ds

    def label_subjects(self, ds):
        """Label the subjects in ds based on .groups

        Parameters
        ----------
        ds : Dataset
            A Dataset with 'subject' entry.
        """
        subject = ds['subject']
        for name, subjects in self.groups.iteritems():
            ds[name] = Var(subject.isin(subjects))

    def load_annot(self, **state):
        """Load a parcellation (from an annot file)

        Returns
        -------
        labels : list of Label
            Labels in the parcellation (output of
            :func:`mne.read_labels_from_annot`).
        """
        self.make_annot(**state)
        return mne.read_labels_from_annot(self.get('mrisubject'),
                                          self.get('parc'), 'both',
                                          subjects_dir=self.get('mri-sdir'))

    def load_bad_channels(self, **kwargs):
        """Load bad channels

        Returns
        -------
        bad_chs : list of str
            Bad chnnels.
        """
        path = self.get('bads-file', **kwargs)
        if os.path.exists(path):
            with open(path) as fid:
                names = [l for l in fid.read().splitlines() if l]
            return names
        else:
            print("No bad channel definition for: %s/%s, creating empty file" %
                  (self.get('subject'), self.get('experiment')))
            self.make_bad_channels(())
            return []

    def _load_cluster_trial_data(self, subject, test, tstart, tstop, pmin,
                                 cluster_pmin=0.05, parc=None, mask=None,
                                 sns_baseline=True, src_baseline=None,
                                 data='src', epoch_inv=None, **kwargs):
        """Load clusters from a test

        Parameters
        ----------
        test, tstart, tstop, pmin
            Test parameters.
        cluster_pmin : float
            Which clusters to load. Use a list of int to specify clusters by
            their ID. Use a float to return all clusters with p-value equal
            or smaller than that value. None to return all clusters (default).
        parc, mask, sns_baseline, src_baseline
            Test parameters.
        """
        if pmin is None or isinstance(pmin, basestring):
            raise NotImplementedError
        with self._temporary_state:
            # load cluster
            res = self.load_test(test, tstart, tstop, pmin, parc, mask, 1,
                                 data, sns_baseline, src_baseline, **kwargs)
            clusters = res.find_clusters(cluster_pmin, True)
            clusters['index'] = clusters.eval("cluster != 0")
            if 'effect' in clusters:
                clusters['name'] = ['%s%s' % (clusters[i, 'effect'].replace(' ', '_'), clusters[i, 'id'])
                                    for i in xrange(clusters.n_cases)]
            else:
                clusters['name'] = ['c%s' % clusters[i, 'id']
                                    for i in xrange(clusters.n_cases)]

            # load single trial data
            if data == 'src' and epoch_inv:
                self.set(inv=epoch_inv)
            subject, group = self._process_subject_arg(subject, {})

            dss = []
            if group is None:
                subjects = (subject,)
            else:
                subjects = self._get_group_members(group)

            for subject in subjects:
                logger.info("Loading single trial data for %s...", subject)
                if data == 'src':
                    ds = self.load_epochs_stc(subject, sns_baseline, src_baseline,
                                              morph=True)
                    y = ds.pop('srcm')
                else:
                    ds = self.load_epochs(subject, sns_baseline)
                    y = ds.pop('sns')

                for c in clusters.itercases():
                    ds[c['name']] = y.mean(c['index'])
                dss.append(ds)
                del y

        return combine(dss)

    def load_cov(self, **kwargs):
        """Load the covariance matrix

        Parameters
        ----------
        others :
            State parameters.
        """
        return mne.read_cov(self.get('cov-file', make=True, **kwargs))

    def load_edf(self, **kwargs):
        """Load the edf file ("edf-file" template)"""
        path = self.get('edf-file', fmatch=False, **kwargs)
        return load.eyelink.Edf(path)

    def _ndvar_name_and_modality(self, ndvar, modality, eog=False):
        """returns (name for the ndvar, data str)"""
        if ndvar:
            if modality == 'meeg':
                raise NotImplementedError("NDVar for sensor space MEEG data")
            elif modality == '':
                data = 'mag'
                if ndvar is True:
                    ndvar = 'meg'
            elif modality == 'eeg':
                if eog:
                    data = 'eeg&eog'
                else:
                    data = 'eeg'

                if ndvar is True:
                    ndvar = 'eeg'
            else:
                raise ValueError("modality=%r" % modality)

            if not isinstance(ndvar, basestring):
                msg = "ndvar needs to be bool or str, got %s" % repr(ndvar)
                raise TypeError(msg)
        else:
            data = None
        return ndvar, data

    def load_epochs(self, subject=None, baseline=None, ndvar=True,
                    add_bads=True, reject=True, add_proj=True, cat=None,
                    decim=None, pad=0, keep_raw=False, eog=False, **kwargs):
        """
        Load a Dataset with epochs for a given epoch definition

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        baseline : None | True | tuple
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to not apply baseline
            correction (None).
        ndvar : bool | str
            Convert epochs to an NDVar with the given name (default is 'meg'
            for MEG data and 'eeg' for EEG data).
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        reject : bool
            Whether to apply epoch rejection or not. The kind of rejection
            employed depends on the :attr:`.epoch_rejection` class attribute.
        add_proj : bool
            Add the projections to the Raw object.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        decim : None | int
            override the epoch decim factor.
        pad : scalar
            Pad the epochs with this much time (in seconds; e.g. for spectral
            analysis).
        keep_raw : bool
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
        eog : bool
            When loading EEG data as NDVar, also add the EOG channels.
        """
        modality = self.get('modality')
        ndvar, data = self._ndvar_name_and_modality(ndvar, modality, eog)
        subject, group = self._process_subject_arg(subject, kwargs)

        if group is not None:
            dss = []
            for _ in self.iter(group=group):
                ds = self.load_epochs(None, baseline, False, add_bads, reject,
                                      add_proj, cat, decim, pad)
                dss.append(ds)

            ds = combine(dss)
        elif modality == 'meeg':  # single subject, combine MEG and EEG
            with self._temporary_state:
                ds_meg = self.load_epochs(subject, baseline, ndvar, add_bads,
                                          reject, add_proj, cat, decim, pad,
                                          False, modality='')
                ds_eeg = self.load_epochs(subject, baseline, ndvar, add_bads,
                                          reject, add_proj, cat, decim, pad,
                                          False, modality='eeg')
            ds, eeg_epochs = align(ds_meg, ds_eeg['epochs'], 'index',
                                   ds_eeg['index'])
            ds['epochs'] = mne.epochs.add_channels_epochs((ds['epochs'], eeg_epochs))
        else:  # single subject, single modality
            epoch = self.epochs[self.get('epoch')]
            if baseline is True:
                baseline = epoch['baseline']

            ds = self.load_selected_events(add_bads=add_bads, reject=reject,
                                           add_proj=add_proj)
            if reject and self._params['rej']['kind'] == 'auto':
                reject_arg = self._params['rej'].get('threshold', None)
            else:
                reject_arg = None

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)

            # load sensor space data
            tmin = epoch['tmin']
            tmax = epoch['tmax']
            if pad:
                tmin -= pad
                tmax += pad
            if decim is None:
                decim = epoch['decim']
            ds = load.fiff.add_mne_epochs(ds, tmin, tmax, baseline, decim=decim,
                                          reject=reject_arg)

            # interpolate channels
            if reject and ds.info[INTERPOLATE_CHANNELS]:
                _interpolate_bads_eeg_epochs(ds['epochs'], ds[INTERPOLATE_CHANNELS])

            if not keep_raw:
                del ds.info['raw']

        if ndvar:
            ds[ndvar] = load.fiff.epochs_ndvar(ds.pop('epochs'), ndvar, data)
            if modality == 'eeg':
                self._fix_eeg_ndvar(ds[ndvar], group)

        return ds

    def load_epochs_stc(self, subject=None, sns_baseline=True,
                        src_baseline=None, ndvar=True, cat=None,
                        keep_epochs=False, morph=False, **kwargs):
        """Load a Dataset with stcs for single epochs

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline : None | True | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction (None).
        ndvar : bool
            Add the source estimates as NDVar named "src" instead of a list of
            SourceEstimate objects named "stc" (default True).
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        keep_epochs : bool
            Keep the sensor space data in the Dataset that is returned (default
            False).
        morph : bool
            Morph the source estimates to the common_brain (default False).
        """
        ds = self.load_epochs(subject, sns_baseline, False, cat=cat, **kwargs)
        self.add_epochs_stc(ds, ndvar, src_baseline, morph)
        if not keep_epochs:
            del ds['epochs']
        return ds

    def load_events(self, subject=None, add_proj=True, add_bads=True,
                    edf=True, **kwargs):
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject : str (state)
            Subject for which to load events.
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        edf : bool
            Loads edf and add it as ``ds.info['edf']``. Edf will only be added
            if ``bool(self.epoch_rejection['edf']) == True``.
        others :
            Update state.
        """
        raw = self.load_raw(add_proj=add_proj, add_bads=add_bads,
                            subject=subject, **kwargs)

        evt_file = self.get('event-file', mkdir=True)

        # search for and check cached version
        ds = None
        if os.path.exists(evt_file):
            event_mtime = os.path.getmtime(evt_file)
            raw_mtime = os.path.getmtime(raw.info['filename'])
            if event_mtime > raw_mtime:
                ds = load.unpickle(evt_file)

        # refresh cache
        subject = self.get('subject')
        if ds is None:
            if self.get('modality') == '':
                merge = -1
            else:
                merge = 0
            ds = load.fiff.events(raw, merge)
            del ds.info['raw']

            if edf and self.has_edf[subject]:  # add edf
                edf = self.load_edf()
                edf.add_t_to(ds)
                ds.info['edf'] = edf

            if edf or not self.has_edf[subject]:
                save.pickle(ds, evt_file)

        ds.info['raw'] = raw
        ds.info['subject'] = subject

        # label events
        a = inspect.getargspec(self.label_events)
        if len(a.args) == 2:
            ds = self.label_events(ds)
        else:
            if a.defaults != (None, None):
                warn("MneExperiment subclasses should remove the subject and "
                     "experiment arguments form the .label_events() method",
                     DeprecationWarning)
            ds = self.label_events(ds, self.get('experiment'), subject)

        if ds is None:
            msg = ("The MneExperiment.label_events() function must return the "
                   "events-Dataset")
            raise RuntimeError(msg)
        return ds

    def load_evoked(self, subject=None, baseline=None, ndvar=True, cat=None,
                    **kwargs):
        """
        Load a Dataset with the evoked responses for each subject.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        baseline : None | True | tuple
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to not apply baseline
            correction (None).
        ndvar : bool
            Convert the mne Evoked objects to an NDVar (the name in the
            Dataset is 'meg' or 'eeg').
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        model : str (state)
            Model according to which epochs are grouped into evoked responses.
        *others* : str
            State parameters.
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        if baseline is True:
            baseline = self.epochs[self.get('epoch')]['baseline']

        if group is not None:
            dss = []
            for _ in self.iter(group=group):
                ds = self.load_evoked(baseline=baseline, ndvar=False, cat=cat)
                dss.append(ds)

            ds = combine(dss)

            # check consistency in MNE objects' number of time points
            lens = [len(e.times) for e in ds['evoked']]
            ulens = set(lens)
            if len(ulens) > 1:
                err = ["Unequal time axis sampling (len):"]
                alens = np.array(lens)
                for l in ulens:
                    err.append('%i: %r' % (l, ds['subject', alens == l].cells))
                raise DimensionMismatchError(os.linesep.join(err))

        else:  # single subject
            ds = self.make_evoked()

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)

            # baseline correction
            if isinstance(baseline, str):
                raise NotImplementedError
            elif baseline:
                for e in ds['evoked']:
                    rescale(e.data, e.times, baseline, 'mean', copy=False)

        # convert to NDVar
        if ndvar:
            modality = self.get('modality')
            if modality == 'meeg':
                modalities = ('meg', 'eeg')
            elif modality == '':
                modalities = ('meg',)
            elif modality == 'eeg':
                modalities = ('eeg',)
            else:
                raise NotImplementedError("modality=%r" % modality)

            for modality in modalities:
                if modality == 'meg':
                    data_arg = 'mag'
                else:
                    data_arg = modality
                ds[modality] = load.fiff.evoked_ndvar(ds['evoked'], None, data_arg)
                if modality == 'eeg':
                    self._fix_eeg_ndvar(ds['eeg'], group)

        return ds

    def load_evoked_freq(self, subject=None, sns_baseline=True,
                         label=None, frequencies='4:40', **kwargs):
        """Load frequency space evoked data

        Parameters
        ----------
        subject : str
            Subject or group.
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        label : None | str | mne.Label
            Label in which to compute the induce power (average in label).
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        model = self.get('model') or None
        if group is not None:
            dss = []
            for _ in self.iter(group=group):
                ds = self.load_evoked_freq(None, sns_baseline, label,
                                           frequencies)
                dss.append(ds)

            ds = combine(dss)
        else:
            if label is None:
                src = self.get('src')
            else:
                src = 'mean'
                if isinstance(label, basestring):
                    label = self.load_label(label)
            ds_epochs = self.load_epochs(None, sns_baseline, False, decim=10, pad=0.2)
            inv = self.load_inv(ds_epochs['epochs'])
            subjects_dir = self.get('mri-sdir')
            ds = source_induced_power('epochs', model, ds_epochs, src, label,
                                      None, inv, subjects_dir, frequencies,
                                      n_cycles=3,
                                      **self._params['apply_inv_kw'])
            ds['subject'] = Factor([subject], repeat=ds.n_cases, random=True)
        return ds

    def load_evoked_stc(self, subject=None, sns_baseline=True,
                        src_baseline=None, sns_ndvar=False, ind_stc=False,
                        ind_ndvar=False, morph_stc=False, morph_ndvar=False,
                        cat=None, keep_evoked=False, **kwargs):
        """Load evoked source estimates.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline : None | True | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction (None).
        ind_stc : bool
            Add source estimates on individual brains as list of
            :class:`mne.SourceEstimate`.
        ind_ndvar : bool
            Add source estimates on individual brains as :class:`NDVar`.
        morph_stc : bool
            Add source estimates morphed to the common brain as list of
            :class:`mne.SourceEstimate`.
        morph_ndvar : bool
            Add source estimates morphed to the common brain as :class:`NDVar`.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        keep_evoked : bool
            Keep the sensor space data in the Dataset that is returned (default
            False).
        *others* : str
            State parameters.
        """
        if not any((ind_stc, ind_ndvar, morph_stc, morph_ndvar)):
            err = ("Nothing to load, set at least one of (ind_stc, ind_ndvar, "
                   "morph_stc, morph_ndvar) to True")
            raise ValueError(err)

        ds = self.load_evoked(subject, sns_baseline, sns_ndvar, cat, **kwargs)
        self.add_evoked_stc(ds, ind_stc, ind_ndvar, morph_stc, morph_ndvar,
                            src_baseline, keep_evoked)

        return ds

    def load_inv(self, fiff=None, **kwargs):
        """Load the inverse operator

        Parameters
        ----------
        fiff : Raw | Epochs | Evoked | ...
            Object which provides the mne info dictionary (default: load the
            raw file).
        others :
            State parameters.
        """
        if self.get('modality', **kwargs) != '':
            raise NotImplementedError("Source reconstruction for EEG data")

        if fiff is None:
            fiff = self.load_raw()

        fwd_file = self.get('fwd-file', make=True)
        fwd = mne.read_forward_solution(fwd_file, surf_ori=True)
        cov = self.load_cov()
        inv = make_inverse_operator(fiff.info, fwd, cov,
                                    **self._params['make_inv_kw'])
        return inv

    def load_label(self, label, **kwargs):
        """Retrieve a label as mne Label object

        Parameters
        ----------
        label : str
            Name of the label. If the label name does not end in '-bh' or '-rh'
            the combination of the labels ``label + '-lh'`` and
            ``label + '-rh'`` is returned.
        """
        labels = self._load_labels(**kwargs)
        if label in labels:
            return labels[label]
        elif not label.endswith(('-lh', '-rh')):
            return labels[label + '-lh'] + labels['-rh']
        else:
            parc = self.get('parc')
            msg = ("Label %r could not be found in parc %r." % (label, parc))
            raise ValueError(msg)

    def _load_labels(self, **kwargs):
        """Load labels from an annotation file."""
        path = self.get('label-file', make=True, **kwargs)
        labels = load.unpickle(path)
        return labels

    def load_morph_matrix(self, **state):
        """Load the morph matrix from mrisubject to common_brain

        Returns
        -------
        mm : sparse matrix
            Morph matrix.
        vertices_to : list of 2 array
            Vertices of the morphed data.
        """
        subjects_dir = self.get('mri-sdir', **state)
        subject_to = self.get('common_brain')
        subject_from = self.get('mrisubject')

        src_to = self.load_src(mrisubject=subject_to, match=False)
        src_from = self.load_src(mrisubject=subject_from, match=False)

        vertices_to = [src_to[0]['vertno'], src_to[1]['vertno']]
        vertices_from = [src_from[0]['vertno'], src_from[1]['vertno']]

        mm = mne.compute_morph_matrix(subject_from, subject_to, vertices_from,
                                      vertices_to, None, subjects_dir)
        return mm, vertices_to

    def load_raw(self, add_proj=True, add_bads=True, preload=False, **kwargs):
        """
        Load a raw file as mne Raw object.

        Parameters
        ----------
        add_proj : bool
            Add the projections to the Raw object. This does *not* set the
            proj variable.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        preload : bool
            Mne Raw parameter.
        """
        if self.get('modality', **kwargs) == 'meeg':
            raise RuntimeError("No raw files for combined MEG & EEG")

        if add_proj:
            proj = self.get('proj')
            if proj:
                proj = self.get('proj-file')
            else:
                proj = None
        else:
            proj = None

        raw = load.fiff.mne_raw(self._get_raw_path(True), proj, preload=preload)
        if add_bads:
            if add_bads is True:
                bad_chs = self.load_bad_channels()
            else:
                bad_chs = add_bads

            raw.info['bads'] = bad_chs

        return raw

    def load_selected_events(self, subject=None, reject=True, add_proj=True,
                             add_bads=True, index=True, **kwargs):
        """
        Load events and return a subset based on epoch and rejection

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a group name
            such as 'all' or a single subject.
        reject : bool | 'keep'
            Reject bad trials. For True, bad trials are removed from the
            Dataset. For 'keep', the 'accept' variable is added to the Dataset
            and bad trials are kept.
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        index : bool | str
            Index the Dataset before rejection (provide index name as str).
        others :
            Update the experiment state.

        Notes
        -----
        When trial rejection is set to automatic, not rejection is performed
        because no epochs are loaded.
        """
        # process arguments
        if reject not in (True, False, 'keep'):
            raise ValueError("Invalid reject value: %r" % reject)
        if index and not isinstance(index, str):
            index = 'index'

        # case of loading events for a group
        subject, group = self._process_subject_arg(subject, kwargs)
        if group is not None:
            dss = [self.load_selected_events(reject=reject, add_proj=add_proj,
                                             add_bads=add_bads, index=index)
                   for _ in self.iter(group=group)]
            ds = combine(dss)
            return ds

        # retrieve & check epoch parameters
        epoch = self.epochs[self.get('epoch')]
        sel = epoch.get('sel', None)
        sel_epoch = epoch.get('sel_epoch', None)
        sub_epochs = epoch.get('sub_epochs', None)
        rej_kind = self._params['rej']['kind']
        if rej_kind not in ('manual', 'make', 'auto'):
            raise ValueError("Unknown rej_kind value: %r" % rej_kind)

        # rejection comes from somewhere else
        if sub_epochs is not None:
            with self._temporary_state:
                dss = [self.load_selected_events(subject, reject, add_proj,
                                                 add_bads, index, epoch=sub_epoch)
                       for sub_epoch in sub_epochs]

                # combine bad channels
                bad_channels = set()
                for ds in dss:
                    if BAD_CHANNELS in ds.info:
                        bad_channels.update(ds.info[BAD_CHANNELS])

                ds = combine(dss)
                ds.info['raw'] = dss[0].info['raw']
                if bad_channels:
                    ds.info[BAD_CHANNELS] = sorted(bad_channels)

            return ds
        elif sel_epoch is not None:
            with self._temporary_state:
                ds = self.load_selected_events(None, 'keep', add_proj, add_bads,
                                               index, epoch=sel_epoch)

            if sel is not None:
                ds = ds.sub(sel)
            if index:
                ds.index(index)

            if reject is True:
                ds = ds.sub('accept')

            return ds

        # load events
        ds = self.load_events(add_proj=add_proj, add_bads=add_bads)
        if sel is not None:
            ds = ds.sub(sel)
        if index:
            ds.index(index)

        n_cases = epoch.get('n_cases', None)
        if n_cases is not None and ds.n_cases != n_cases:
            err = "Number of epochs %i, expected %i" % (ds.n_cases, n_cases)
            raise RuntimeError(err)

        # rejection
        if not reject or rej_kind == '':
            pass
        elif rej_kind in ('manual', 'make'):
            path = self.get('rej-file')
            if not os.path.exists(path):
                err = ("The rejection file at %r does not exist. Run "
                       ".make_rej() first." % path)
                raise RuntimeError(err)

            # load and check file
            ds_sel = load.unpickle(path)
            if not np.all(ds['trigger'] == ds_sel['trigger']):
                if np.all(ds[:-1, 'trigger'] == ds_sel['trigger']):
                    ds = ds[:-1]
                    msg = self.format("Last epoch for {subject} is missing")
                    logger.warn(msg)
                else:
                    err = ("The epoch selection file contains different "
                           "events than the data. Something went wrong...")
                    raise RuntimeError(err)

            interpolate_channels = self._params['rej']['interpolation']
            if interpolate_channels and self.get('modality') == 'eeg' and INTERPOLATE_CHANNELS in ds_sel:
                ds[INTERPOLATE_CHANNELS] = ds_sel[INTERPOLATE_CHANNELS]
                ds.info[INTERPOLATE_CHANNELS] = True
            else:
                ds.info[INTERPOLATE_CHANNELS] = False

            # subset events
            if reject == 'keep':
                ds['accept'] = ds_sel['accept']
            elif reject is True:
                ds = ds.sub(ds_sel['accept'])
            else:
                err = ("reject parameter must be bool or 'keep', not "
                       "%r" % reject)
                raise ValueError(err)

            # bad channels
            if add_bads and BAD_CHANNELS in ds_sel.info:
                ds.info[BAD_CHANNELS] = ds_sel.info[BAD_CHANNELS]
        else:
            use = self._params['rej'].get('edf', False)
            if use:
                edf = ds.info['edf']
                tmin = epoch.get('reject_tmin', epoch['tmin'])
                tmax = epoch.get('reject_tmax', epoch['tmax'])
                if reject == 'keep':
                    edf.mark(ds, tstart=tmin, tstop=tmax, use=use)
                else:
                    ds = edf.filter(ds, tstart=tmin, tstop=tmax, use=use)

        return ds

    def load_src(self, add_geom=False, **state):
        "Load the current source space"
        fpath = self.get('src-file', make=True, **state)
        src = mne.read_source_spaces(fpath, add_geom)
        return src

    def load_test(self, test, tstart, tstop, pmin, parc=None, mask=None,
                  samples=1000, data='src', sns_baseline=True,
                  src_baseline=None, return_data=False, make=False, redo=False,
                  **kwargs):
        """Create and load spatio-temporal cluster test results

        Parameters
        ----------
        test : None | str
            Test for which to create a report (entry in MneExperiment.tests;
            None to use the test that was specified most recently).
        tstart, tstop : None | scalar
            Time window for finding clusters.
        pmin : float | 'tfce' | None
            Kind of test.
        parc : None | str
            Parcellation for which to collect distribution.
        mask : None | str
            Mask whole brain.
        samples : int
            Number of samples used to determine cluster p values for spatio-
            temporal clusters.
        data : 'sns' | 'src'
            Whether the analysis is in sensor or source space.
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline : None | True | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction (None).
        return_data : bool
            Return the data along with the test result (see below).
        make : bool
            If the target file does not exist, create it (could take a long
            time depending on the test; if False, raise an IOError).
        redo : bool
            If the target file already exists, delete and recreate it (only
            applies for tests that are cached).
        group : str
            Group for which to perform the test.

        Returns
        -------
        ds : Dataset (if return_data==True)
            Data that forms the basis of the test.
        res : TestResult
            Test result for the specified test.
        """
        self._set_test_options(data, sns_baseline, src_baseline, pmin, tstart,
                               tstop)

        # figure out what test to do
        if test is None:
            test = self.get('test', **kwargs)
        else:
            self.set(test=test, **kwargs)

        if self._tests[test]['kind'] == 'custom':
            raise RuntimeError("Don't know how to perform 'custom' test")

        # find data to use
        modality = self.get('modality')
        if data == 'sns':
            if modality == '':
                y_name = 'meg'
            elif modality == 'eeg':
                y_name = 'eeg'
            else:
                raise ValueError("data=%r, modality=%r" % (data, modality))
        elif data == 'src':
            y_name = 'srcm'
        else:
            raise ValueError("data=%s" % repr(data))

        # find cached file path
        if parc:
            if pmin == 'tfce':
                raise NotImplementedError("tfce analysis can't have parc")
            elif data == 'sns':
                raise NotImplementedError("sns analysis can't have parc")
            mask = True
            parc_ = parc
            parc_dim = 'source'
            data_parc = parc
        elif mask:
            if data == 'sns':
                raise NotImplementedError("sns analysis can't have mask")
            parc_ = mask
            if pmin is None:  # can as well collect dist for on parc
                parc_dim = 'source'
                data_parc = mask
            else:  # parc means disconnecting
                parc_dim = None
                data_parc = '%s-mask' % mask
        else:
            parc_ = 'aparc'
            parc_dim = None
            data_parc = 'unmasked'

        dst = self.get('test-file', mkdir=True, data_parc=data_parc,
                       parc=parc_)

        # try to load cached test
        if not redo and os.path.exists(dst):
            res = load.unpickle(dst)
            if res.samples >= samples or res.samples == -1:
                load_data = return_data
            elif make:
                res = None
                load_data = True
            else:
                msg = ("The cached test was performed with fewer samples than "
                       "requested (%i vs %i). Set a lower number of samples, "
                       "or set make=True to perform the test with the higher "
                       "number of samples." % (res.samples, samples))
                raise IOError(msg)
        elif redo or make:
            res = None
            load_data = True
        else:
            msg = ("The requested test is not cached. Set make=True to "
                   "perform the test.")
            raise IOError(msg)

        # load data
        if load_data:
            # determine categories to load
            test_params = self._tests[test]
            if test_params['kind'] == 'ttest_rel':
                cat = (test_params['c1'], test_params['c0'])
            else:
                cat = None

            # load data
            if data == 'sns':
                ds = self.load_evoked(True, sns_baseline, True, cat)
            elif data == 'src':
                ds = self.load_evoked_stc(True, sns_baseline, src_baseline,
                                          morph_ndvar=True, cat=cat)
                if mask:
                    # reduce data to parc
                    y = ds['srcm']
                    idx_masked = y.source.parc.startswith('unknown')
                    if np.any(idx_masked):
                        idx = np.invert(idx_masked)
                        ds['srcm'] = y.sub(source=idx)

        # perform the test if it was not cached
        if res is None:
            res = self._make_test(ds[y_name], ds, test, samples, pmin, tstart,
                                  tstop, None, parc_dim)
            # cache
            save.pickle(res, dst)

        if return_data:
            return ds, res
        else:
            return res

    def make_annot(self, redo=False, **state):
        """Make sure the annot files for both hemispheres exist

        Parameters
        ----------
        redo : bool
            Even if the file exists, recreate it (default False).

        Returns
        -------
        mtime : float | None
            Modification time of the existing files, or None if they were newly
            created.
        """
        self.set(**state)

        # variables
        parc = self.get('parc')
        if parc == '':
            return
        mrisubject = self.get('mrisubject')
        common_brain = self.get('common_brain')

        if parc in FS_PARCS:
            if redo or self._annot_mtime() is None:
                raise NotImplemented("The Aparc FreeSurfer parcellations can "
                                     "not be created autmaticaly yet. Use "
                                     "FreeSurfer to create the %s "
                                     "parcellation for mrisubject %s."
                                     % (parc, mrisubject))
                # https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
        elif mrisubject == common_brain:
            # check existing files
            if not redo:
                mtime = self._annot_mtime()
                if mtime is not None:
                    return mtime
            # make sure it's not one that should exist
            if parc in FSA_PARCS:
                raise RuntimeError("The %s parcellation is missing from your "
                                   "common_brain" % parc)
            # create it
            labels = self._make_annot(parc, mrisubject)
            mri_sdir = self.get('mri-sdir')
            write_labels_to_annot(labels, mrisubject, parc, True, mri_sdir)
        else:
            # make sure annot exists for common brain
            self.set(mrisubject=common_brain, match=False)
            common_brain_mtime = self.make_annot()
            self.set(mrisubject=mrisubject, match=False)

            # check whether subject's file is current
            if not redo and common_brain_mtime is not None:
                mtime = self._annot_mtime()
                if mtime is not None and mtime > common_brain_mtime:
                    return mtime

            # copy or morph
            if is_fake_mri(self.get('mri-dir')):
                for _ in self.iter('hemi'):
                    self.make_copy('annot-file', 'mrisubject', common_brain,
                                   mrisubject)
            else:
                self.get('label-dir', make=True)
                subjects_dir = self.get('mri-sdir')
                for hemi in ('lh', 'rh'):
                    cmd = ["mri_surf2surf", "--srcsubject", common_brain,
                           "--trgsubject", mrisubject, "--sval-annot", parc,
                           "--tval", parc, "--hemi", hemi]
                    subp.run_freesurfer_command(cmd, subjects_dir)
                fix_annot_names(mrisubject, parc, common_brain,
                                subjects_dir=subjects_dir)

    def _make_annot(self, parc, subject):
        """Returns labels

        Notes
        -----
        Only called to make custom annotation files for the common_brain
        """
        if parc == 'lobes':
            sdir = self.get('mri-sdir')

            # load source annot
            with self._temporary_state:
                labels = self.load_annot(parc='PALS_B12_Lobes')

            # sort labels
            labels = [l for l in labels if l.name[:-3] != 'MEDIAL.WALL']

            # rename good labels
            rename_label(labels, 'LOBE.FRONTAL', 'frontal')
            rename_label(labels, 'LOBE.OCCIPITAL', 'occipital')
            rename_label(labels, 'LOBE.PARIETAL', 'parietal')
            rename_label(labels, 'LOBE.TEMPORAL', 'temporal')

            # reassign unwanted labels
            targets = ('frontal', 'occipital', 'parietal', 'temporal')
            dissolve_label(labels, 'LOBE.LIMBIC', targets, sdir)
            dissolve_label(labels, 'GYRUS', targets, sdir, 'rh')
            dissolve_label(labels, '???', targets, sdir)
            dissolve_label(labels, '????', targets, sdir, 'rh')
            dissolve_label(labels, '???????', targets, sdir, 'rh')
        elif parc == 'lobes-op':
            with self._temporary_state:
                lobes = {l.name: l for l in self.load_annot(parc='lobes')}
            labels = [lobes['occipital-lh'] + lobes['parietal-lh'],
                      lobes['occipital-rh'] + lobes['parietal-rh']]
        elif parc == 'lobes-ot':
            with self._temporary_state:
                lobes = {l.name: l for l in self.load_annot(parc='lobes')}
            labels = [lobes['occipital-lh'] + lobes['temporal-lh'],
                      lobes['occipital-rh'] + lobes['temporal-rh']]
        else:
            msg = ("At least one of the annot files for the custom parcellation "
                   "%r is missing for %r, and a make function is not "
                   "implemented." % (parc, subject))
            raise NotImplementedError(msg)
        return labels

    def make_bad_channels(self, bad_chs, redo=False, **kwargs):
        """Write the bad channel definition file for a raw file

        Parameters
        ----------
        bad_chs : iterator of str
            Names of the channels to set as bad. Numerical entries are
            interpreted as "MEG XXX". If bad_chs contains entries not present
            in the raw data, a ValueError is raised.
        redo : bool
            If the file already exists, replace it.

        See Also
        --------
        load_bad_channels : load the current bad_channels file
        """
        dst = self.get('bads-file', **kwargs)
        if os.path.exists(dst):
            old_bads = self.load_bad_channels()
            if not redo:
                msg = ("Bads file already exists with %s. In order to replace "
                       "it, use `redo=True`." % old_bads)
                raise IOError(msg)
        else:
            old_bads = None

        raw = self.load_raw(add_bads=False)
        sensor = load.fiff.sensor_dim(raw)
        chs = sensor._normalize_sensor_names(bad_chs)
        if old_bads is None:
            print "-> %s" % chs
        else:
            print "%s -> %s" % (old_bads, chs)
        text = os.linesep.join(chs)
        with open(dst, 'w') as fid:
            fid.write(text)

    def make_besa_evt(self, redo=False, **state):
        """Make the trigger and event files needed for besa

        Parameters
        ----------
        redo : bool
            If besa files already exist, overwrite them.

        Notes
        -----
        Ignores the *decim* epoch parameter.

        Target files are saved relative to the *besa-root* location.
        """
        self.set(**state)
        rej = self.get('rej')
        trig_dest = self.get('besa-trig', rej='', mkdir=True)
        evt_dest = self.get('besa-evt', rej=rej, mkdir=True)
        if not redo and os.path.exists(evt_dest) and os.path.exists(trig_dest):
            return

        # load events
        ds = self.load_selected_events(reject='keep')

        # save triggers
        if redo or not os.path.exists(trig_dest):
            save.meg160_triggers(ds, trig_dest, pad=1)
            if not redo and os.path.exists(evt_dest):
                return
        else:
            ds.index('besa_index', 1)

        # reject bad trials
        ds = ds.sub('accept')

        # save evt
        epoch = self.epochs[self.get('epoch')]
        save.besa_evt(ds, tstart=epoch['tmin'], tstop=epoch['tmax'], dest=evt_dest)

    def make_copy(self, temp, field, src, dst, redo=False):
        """Make a copy of a file

        Parameters
        ----------
        temp : str
            Template of the file which to copy.
        field : str
            Field in which the source and target of the link are distinguished.
        src : str
            Value for field on the source file.
        dst : str
            Value for field on the destination filename.
        redo : bool
            If the target file already exists, overwrite it.
        """
        dst_path = self.get(temp, mkdir=True, **{field: dst})
        if not redo and os.path.exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        if os.path.isdir(src_path):
            raise ValueError("Can only copy files, not directories.")
        shutil.copyfile(src_path, dst_path)

    def make_cov(self, redo=False):
        """Make a noise covariance (cov) file

        Parameters
        ----------
        redo : bool
            If the cov file already exists, overwrite it.
        """
        dest = self.get('cov-file')
        if (not redo) and os.path.exists(dest):
            cov_mtime = os.path.getmtime(dest)
            raw_mtime = os.path.getmtime(self._get_raw_path())
            bads_mtime = os.path.getmtime(self.get('bads-file'))
            if cov_mtime > max(raw_mtime, bads_mtime):
                return

        rej = self.get('cov-rej')
        params = self._covs[self.get('cov')]
        epoch = params.get('epoch', 'cov')
        method = params.get('method', 'empirical')
        keep_sample_mean = params.get('keep_sample_mean', True)
        reg = params.get('reg', None)

        with self._temporary_state:
            ds = self.load_epochs(None, True, False, decim=1, epoch=epoch, rej=rej)
        epochs = ds['epochs']
        if has_mne_09:
            cov = mne.compute_covariance(epochs, keep_sample_mean, method=method)
        elif method == 'empirical':
            cov = mne.compute_covariance(epochs, keep_sample_mean)
        else:
            raise RuntimeError("Cov with method=%r requires mne >= 0.9" % method)

        if reg is True:
            cov = mne.cov.regularize(cov, epochs.info)
        elif isinstance(reg, dict):
            cov = mne.cov.regularize(cov, epochs.info, **reg)
        elif reg == 'best':
            if mne.pick_types(epochs.info, meg='grad', eeg=True, ref_meg=False):
                raise NotImplementedError("EEG or gradiometer sensors")
            reg_vs = np.arange(0, 0.21, 0.01)
            covs = [mne.cov.regularize(cov, epochs.info, mag=v) for v in reg_vs]

            # compute whitened global field power
            evoked = epochs.average()
            picks = mne.pick_types(evoked.info, meg='mag', ref_meg=False)
            gfps = [mne.whiten_evoked(evoked, cov, picks).data.std(0)
                    for cov in covs]

            # apply padding
            t_pad = params.get('reg_eval_win_pad', 0)
            if t_pad:
                n_pad = int(t_pad * epochs.info['sfreq'])
                if len(gfps[0]) <= 2 * n_pad:
                    msg = "Covariance padding (%s) is bigger than epoch" % t_pad
                    raise ValueError(msg)
                padding = slice(n_pad, -n_pad)
                gfps = [gfp[padding] for gfp in gfps]

            vs = [gfp.mean() for gfp in gfps]
            i = np.argmin(np.abs(1 - np.array(vs)))
            cov = covs[i]

            # save cov value
            with open(self.get('cov-info-file'), 'w') as fid:
                fid.write('%s\n' % reg_vs[i])

        cov.save(dest)

    def make_evoked(self, redo=False, **kwargs):
        """
        Creates datasets with evoked sensor data.

        Parameters
        ----------
        epoch : str
            Data epoch specification.
        model : str
            Model specifying cells for evoked.
        """
        dest = self.get('evoked-file', mkdir=True, **kwargs)
        if not redo and os.path.exists(dest):
            evoked_mtime = os.path.getmtime(dest)
            raw_mtime = os.path.getmtime(self._get_raw_path(make=True))
            bads_mtime = os.path.getmtime(self.get('bads-file'))

            rej_file_epochs = self.epochs[self.get('epoch')].get('_rej_file_epochs', None)
            if rej_file_epochs is None:
                sel_file = self.get('rej-file')
                rej_mtime = os.path.getmtime(sel_file)
            else:
                with self._temporary_state:
                    paths = [self.get('rej-file', epoch=e) for e in rej_file_epochs]
                rej_mtime = max(map(os.path.getmtime, paths))

            if evoked_mtime > max(raw_mtime, bads_mtime, rej_mtime):
                ds = load.unpickle(dest)
                if ds.info.get('mne_version', None) == mne.__version__:
                    return ds

        # load the epochs
        ds = self.load_epochs(ndvar=False)

        # aggregate
        equal_count = self.get('equalize_evoked_count') == 'eq'
        ds_agg = ds.aggregate(self.get('model'), drop_bad=True,
                              drop=('i_start', 't_edf', 'T', 'index'),
                              equal_count=equal_count, never_drop=('epochs',))

        # save
        ds_agg.rename('epochs', 'evoked')
        ds_agg.info['mne_version'] = mne.__version__
        if 'raw' in ds_agg.info:
            del ds_agg.info['raw']
        save.pickle(ds_agg, dest)
        return ds_agg

    def make_fwd(self, redo=False):
        """Make the forward model"""
        dst = self.get('fwd-file')
        raw = self._get_raw_path(make=True)
        trans = self.get('trans-file')
        src = self.get('src-file', make=True)
        bem = self.get('bem-sol-file', fmatch=True)

        if not redo and os.path.exists(dst):
            fwd_mtime = os.path.getmtime(dst)
            raw_mtime = os.path.getmtime(raw)
            trans_mtime = os.path.getmtime(trans)
            src_mtime = os.path.getmtime(src)
            bem_mtime = os.path.getmtime(bem)
            if fwd_mtime > max(raw_mtime, trans_mtime, src_mtime, bem_mtime):
                return

        if self.get('modality') != '':
            raise NotImplementedError("Source reconstruction with EEG")
        src = mne.read_source_spaces(src)
        fwd = mne.make_forward_solution(raw, trans, src, bem, ignore_ref=True)
        for s, s0 in izip(fwd['src'], src):
            if s['nuse'] != s0['nuse']:
                msg = ("The forward solution contains fewer sources than the "
                       "source space. This could be due to a corrupted bem "
                       "file with source outside the inner skull surface.")
                raise RuntimeError(msg)
        mne.write_forward_solution(dst, fwd, True)

    def make_labels(self, redo=False):
        dst = self.get('label-file', mkdir=True)
        if not redo and os.path.exists(dst):
            return
        elif redo:
            self.make_annot(redo)

        labels = self.load_annot()
        label_dict = {label.name: label for label in labels}
        save.pickle(label_dict, dst)

    def make_link(self, temp, field, src, dst, redo=False):
        """Make a hard link at the file with the dst value on field, linking to
        the file with the src value of field

        Parameters
        ----------
        temp : str
            Template of the file for which to make a link.
        field : str
            Field in which the source and target of the link are distinguished.
        src : str
            Value for field on the source file.
        dst : str
            Value for field on the destination filename.
        redo : bool
            If the target file already exists, overwrite it.
        """
        dst_path = self.get(temp, **{field: dst})
        if not redo and os.path.exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        os.link(src_path, dst_path)

    def make_mov_ga_dspm(self, subject=None, surf='inflated', fmin=2,
                         redo=False, **kwargs):
        """Make a grand average movie from dSPM values

        Parameters
        ----------
        subject : None | str
            Subject or group.
        surf : str
            Surface on which to plot data.
        fmin : scalar
            Minimum dSPM value to draw. fmax is 3 * fmin.
        redo : bool
            Make the movie even if the target file exists already.
        others :
            Experiment state parameters.
        """
        kwargs['model'] = ''
        subject, group = self._process_subject_arg(subject, kwargs)

        self.set(equalize_evoked_count='',
                 analysis='{src-kind} {evoked-kind}',
                 resname="GA dSPM %s %s" % (surf, fmin),
                 ext='mov')
        if group is not None:
            dst = self.get('res-g-file', mkdir=True)
            src = 'srcm'
        else:
            dst = self.get('res-s-file', mkdir=True)
            src = 'src'
        if not redo and os.path.exists(dst):
            return

        ds = self.load_evoked_stc(group, ind_ndvar=subject is not None,
                                  morph_ndvar=group is not None)

        brain = plot.brain.dspm(ds[src], fmin, fmin * 3, surf=surf)
        brain.save_movie(dst)
        brain.close()

    def make_mov_ga(self, subject=None, surf='smoothwm', p0=0.05, redo=False,
                    **kwargs):
        """Make a grand average movie for a subject or group

        In order to compare activation with baseline, data are not baseline
        corrected in sensor space but in source space.

        Parameters
        ----------
        subject : None | str
            Subject or group.
        surf : str
            Surface on which to plot data.
        p0 : scalar
            Minimum p value to draw.
        redo : bool
            Make the movie even if the target file exists already.
        others :
            Experiment state parameters.
        """
        subject, group = self._process_subject_arg(subject, kwargs)

        if p0 == 0.05:
            p1 = 0.01
        elif p0 == 0.01:
            p1 = 0.001
        else:
            raise ValueError("Unknown p0: %s" % p0)

        self.set(equalize_evoked_count='',
                 analysis='{src-kind} {evoked-kind}',
                 resname="GA %s %s" % (surf, p0),
                 ext='mov')
        if group is not None:
            dst = self.get('res-g-file', mkdir=True)
        else:
            dst = self.get('res-s-file', mkdir=True)
        if not redo and os.path.exists(dst):
            return

        inv = self.get('inv')
        if inv.startswith(('free', 'loose')):
            sns_baseline = None
            src_baseline = self.epochs[self.get('epoch')]['baseline']
        elif inv.startswith('fixed'):
            sns_baseline = self.epochs[self.get('epoch')]['baseline']
            src_baseline = None
        else:
            raise ValueError("Unknown inv kind: %r" % inv)

        self.set(model='')
        if group is not None:
            ds = self.load_evoked_stc(group, sns_baseline, src_baseline,
                                      morph_ndvar=True)
            src = 'srcm'
        else:
            ds = self.load_epochs_stc(group, sns_baseline, src_baseline)
            src = 'src'

        res = testnd.ttest_1samp(src, match=None, ds=ds)
        brain = plot.brain.stat(res.p, res.t, p0=p0, p1=p1, surf=surf,
                                dtmin=0.01)
        brain.save_movie(dst)
        brain.close()

    def make_mov_ttest(self, x, c1, c0, subject=None, surf='inflated', p0=0.05,
                       redo=False, **kwargs):
        """Make a t-test movie

        Parameters
        ----------
        x : str
            Model on which the conditions c1 and c0 are defined.
        c1 : None | str | tuple
            Test condition (cell in model). If None, the grand average is
            used and c0 has to be a scalar.
        c0 : str | scalar
            Control condition (cell on model) or scalar against which to
            compare c1.
        subject : str (state)
            Group name, or subject name for single subject ttest.
        """
        subject, group = self._process_subject_arg(subject, kwargs)

        if p0 == 0.1:
            p1 = 0.05
        elif p0 == 0.05:
            p1 = 0.01
        elif p0 == 0.01:
            p1 = 0.001
        else:
            raise ValueError("Unknown p0: %s" % p0)

        self.set(analysis='{src-kind} {evoked-kind}',
                 resname="T-Test %s-%s %s %s" % (str(c1), str(c0), surf, p0),
                 ext='mov')
        if group is not None:
            dst = self.get('res-g-file', mkdir=True)
        else:
            dst = self.get('res-s-file', mkdir=True)
        if not redo and os.path.exists(dst):
            return

        sns_baseline = self.epochs[self.get('epoch')]['baseline']
        src_baseline = None

        if group is not None:
            ds = self.load_evoked_stc(group, sns_baseline, src_baseline,
                                      morph_ndvar=True, cat=(c1, c0), model=x)
            res = testnd.ttest_rel('srcm', x, c1, c0, match='subject', ds=ds)
        else:
            ds = self.load_epochs_stc(group, sns_baseline, src_baseline,
                                      cat=(c1, c0), model=x)
            res = testnd.ttest_ind('src', x, c1, c0, ds=ds)

        brain = plot.brain.stat(res.p, res.t, p0=p0, p1=p1, surf=surf,
                                dtmin=0.01)
        brain.save_movie(dst)
        brain.close()

    def make_mrat_evoked(self, **kwargs):
        """Produce the sensor data fiff files needed for MRAT sensor analysis

        Parameters
        ----------
        kwargs :
            State arguments

        Examples
        --------
        To produce evoked files for all subjects in the experiment:

        >>> experiment.set(model='factor1%factor2')
        >>> for _ in experiment:
        >>>     experiment.make_mrat_evoked()
        ...
        """
        ds = self.load_evoked(ndvar=False, **kwargs)

        # create fiffs
        model = self.get('model')
        factors = [f.strip() for f in model.split('%')]
        for case in ds.itercases():
            condition = '_'.join(case[f] for f in factors)
            path = self.get('mrat-sns-file', mkdir=True,
                            mrat_condition=condition)
            evoked = case['evoked']
            evoked.save(path)

    def make_mrat_stcs(self, **kwargs):
        """Produce the STC files needed for the MRAT analysis tool

        Parameters
        ----------
        kwargs :
            State arguments

        Examples
        --------
        To produce stc files for all subjects in the experiment:

        >>> experiment.set_inv('free')
        >>> experiment.set(model='factor1%factor2')
        >>> for _ in experiment:
        >>>     experiment.make_mrat_stcs()
        ...
        """
        ds = self.load_evoked_stc(morph_stc=True, **kwargs)

        # save condition info
        info_file = self.get('mrat_info-file', mkdir=True)
        ds.save_txt(info_file)

        # create stcs
        model = self.get('model')
        factors = [f.strip() for f in model.split('%')]
        for case in ds.itercases():
            condition = '_'.join(case[f] for f in factors)
            path = self.get('mrat-src-file', mkdir=True,
                            mrat_condition=condition)
            stc = case['stcm']
            stc.save(path)

    def make_plot_annot(self, surf='inflated', redo=False, **state):
        """Create a figure for the contents of an annotation file

        Parameters
        ----------
        surf : str
            FreeSurfer surface on which to plot the annotation.
        redo : bool
            If the target file already exists, overwrite it.
        """
        mrisubject = self.get('mrisubject', **state)
        if is_fake_mri(self.get('mri-dir')):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)

        analysis = 'Source Annot'
        resname = "{parc} {mrisubject} %s" % surf
        ext = 'png'
        dst = self.get('res-file', mkdir=True, analysis=analysis,
                       resname=resname, ext=ext)
        if not redo and os.path.exists(dst):
            return

        brain = self.plot_annot(surf, w=1200)
        brain.save_image(dst)

    def make_plot_label(self, label, surf='inflated', redo=False, **state):
        if is_fake_mri(self.get('mri-dir', **state)):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)

        dst = self._make_plot_label_dst(surf, label)
        if not redo and os.path.exists(dst):
            return

        brain = self.plot_label(label, surf=surf)
        brain.save_image(dst)

    def make_plots_labels(self, surf='inflated', redo=False, **state):
        self.set(**state)
        self.store_state()
        if is_fake_mri(self.get('mri-dir')):
            self.set(mrisubject=self.get('common_brain'), match=False)

        labels = self._load_labels().values()
        dsts = [self._make_plot_label_dst(surf, label.name)
                for label in labels]
        if not redo and all(os.path.exists(dst) for dst in dsts):
            return

        brain = self.plot_brain(surf, None, 'split', ['lat', 'med'], w=1200)
        for label, dst in zip(labels, dsts):
            brain.add_label(label)
            brain.save_image(dst)
            brain.remove_labels(hemi='lh')
        self.restore_state()

    def _make_plot_label_dst(self, surf, label):
        analysis = 'Source Labels'
        folder = "{parc} {mrisubject} %s" % surf
        resname = label
        ext = 'png'
        dst = self.get('res-deep-file', mkdir=True, analysis=analysis,
                       folder=folder, resname=resname, ext=ext)
        return dst

    def make_proj(self, save=True, save_plot=True):
        """
        computes the first ``n_mag`` PCA components, plots them, and asks for
        user input (a tuple) on which ones to save.

        Parameters
        ----------
        epochs : mne.Epochs
            epochs which should be used for the PCA
        dest : str(path)
            path where to save the projections
        n_mag : int
            number of components to compute
        save : False | True | tuple
            False: don'r save proj fil; True: manuall pick componentws to
            include in the proj file; tuple: automatically include these
            components
        save_plot : False | str(path)
            target path to save the plot
        """
        proj = self.get('proj')
        opt = self.projs[proj]
        reject = {'mag': 1.5e-11}  # far lower than the data range
        if opt['base'] == 'raw':
            raw = self.load_raw(add_proj=False)

            # select time range of events
            events = load.fiff.events(raw)
            time = events['i_start'] / events.info['samplingrate']
            tstart = time.x.min() - 5
            tstop = time.x.max() + 5

            projs = mne.compute_proj_raw(raw, tstart, tstop, duration=5,
                                         n_grad=0, n_mag=9, n_eeg=0,
                                         reject=reject, n_jobs=1)
            fiff_obj = raw
        else:
            epoch = opt['base']
            rej = opt.get('rej', None)
            rej_ = '' if rej is None else rej
            self.set(epoch=epoch, rej=rej_)
            ds = self.load_epochs(ndvar=False, add_proj=False)
            epochs = ds['epochs']
            if rej is None:
                epochs.reject = reject.copy()
                epochs._reject_setup()
            projs = mne.compute_proj_epochs(epochs, n_grad=0, n_mag=9, n_eeg=0)
            fiff_obj = epochs

        # convert projs to NDVar
        picks = mne.epochs.pick_types(fiff_obj.info, exclude='bads')
        sensor = load.fiff.sensor_dim(fiff_obj, picks=picks)
        projs_ndvars = []
        for p in projs:
            d = p['data']['data'][0]
            name = p['desc'][-5:]
            v = NDVar(d, (sensor,), name=name)
            projs_ndvars.append(v)

        # plot PCA components
        proj_file = self.get('proj-file')
        p = plot.Topomap(projs_ndvars, title=proj_file, ncol=3, w=9)
        if save_plot:
            dest = self.get('plot-file', analysis='proj', ext='pdf',
                            name='{subject}_{experiment}_{raw-kind}')
            p.figure.savefig(dest)
        if save:
            rm = save
            title = "Select Projections"
            msg = ("which Projections do you want to select? (tuple / 'x' to "
                   "abort)")
            while not isinstance(rm, tuple):
                answer = ui.ask_str(msg, title, default='(0,)')
                rm = eval(answer)
                if rm == 'x': raise

            p.close()
            projs = [projs[i] for i in rm]
            mne.write_proj(proj_file, projs)

    def make_raw(self, redo=False, n_jobs=1, **kwargs):
        """Make a raw file

        Parameters
        ----------
        raw : str
            Name of the raw file to make.
        redo : bool
            If the file already exists, recreate it.
        n_jobs : int
            Number of processes for multiprocessing.

        Notes
        -----
        Due to the electronics of the KIT system sensors, signal lower than
        0.16 Hz is not recorded even when recording at DC.
        """
        raw_dst = self.get('raw', **kwargs)
        if self._raw[raw_dst] is None:
            raise RuntimeError("Can't make %r raw file because it is an input "
                               "file" % raw_dst)
        dst = self.get('cached-raw-file', mkdir=True)
        with self._temporary_state:
            if not redo and os.path.exists(dst):
                src = self.get('raw-file', raw='clm')
                src_mtime = os.path.getmtime(src)
                dst_mtime = os.path.getmtime(dst)
                if dst_mtime > src_mtime:
                    return

            apply_proj = False
            raw = self.load_raw(raw='clm', add_proj=apply_proj, add_bads=False,
                                preload=True)

        if apply_proj:
            raw.apply_projector()

        args, kwargs = self._raw[raw_dst]
        raw.filter(*args, n_jobs=n_jobs, **kwargs)
        raw.save(dst, overwrite=True)

    def make_rej(self, **kwargs):
        """Show the SelectEpochs GUI to do manual epoch selection for a given
        epoch

        The GUI is opened with the correct file name; if the corresponding
        file exists, it is loaded, and upon saving the correct path is
        the default.

        Parameters
        ----------
        kwargs :
            Kwargs for SelectEpochs
        """
        rej_args = self._params['rej']
        if not rej_args['kind'] == 'manual':
            err = ("Epoch rejection kind for rej=%r is not manual. See the "
                   ".epoch_rejection class attribute." % self.get('rej'))
            raise RuntimeError(err)

        epoch = self.epochs[self.get('epoch')]
        if 'sel_epoch' in epoch:
            msg = ("The current epoch {cur!r} inherits rejections from "
                   "{sel!r}. To access a rejection file for this epoch, call "
                   "`e.set(epoch={sel!r})` and then `e.make_rej()` "
                   "again.".format(cur=epoch['name'], sel=epoch['sel_epoch']))
            raise ValueError(msg)

        path = self.get('rej-file', mkdir=True)
        modality = self.get('modality')

        if modality == '':
            ds = self.load_epochs(reject=False, eog=True,
                                  decim=rej_args.get('decim', None))
            subject = self.get('subject')
            subject_prefix = self._subject_re.match(subject).group(1)
            meg_system = self._meg_systems[subject_prefix]
            eog_sns = self._eog_sns[meg_system]
            data = 'meg'
            vlim = 2e-12
            allow_interpolation = False
        elif modality == 'eeg':
            ds = self.load_epochs(reject=False, eog=True, baseline=True,
                                  decim=rej_args.get('decim', None))
            eog_sns = self._eog_sns['KIT-BRAINVISION']
            data = 'eeg'
            vlim = 1.5e-4
            allow_interpolation = rej_args['interpolation']
        else:
            raise ValueError("modality=%r" % modality)

        # don't mark eog sns if it is bad
        bad_channels = self.load_bad_channels()
        eog_sns = [c for c in eog_sns if c not in bad_channels]

        gui.select_epochs(ds, data, path=path, vlim=vlim, mark=eog_sns,
                          allow_interpolation=allow_interpolation, **kwargs)

    def make_report(self, test, parc=None, mask=None, pmin=None, tstart=0.15,
                    tstop=None, samples=10000, sns_baseline=True,
                    src_baseline=None, include=0.2, redo=False,
                    redo_test=False, **state):
        """Create an HTML report on spatio-temporal clusters

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        parc : None | str
            Find clusters in each label of parc (as opposed to the whole
            brain).
        mask : None | str
            Parcellation to apply as mask. Can only be specified if parc==None.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline : None | True | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction (None).
        include : 0 < scalar <= 1
            Create plots for all clusters with p-values smaller or equal this value.
        redo : bool
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        redo_test : bool
            Redo the test even if a cached file exists.
        """
        if samples < 1:
            raise ValueError("samples needs to be > 0")

        if include <= 0 or include > 1:
            raise ValueError("include needs to be 0 < include <= 1, got %s"
                             % repr(include))

        # determine report file name
        if parc is None:
            if mask:
                folder = "%s Masked" % mask.capitalize()
            else:
                folder = "Whole Brain"
        elif mask:
            raise ValueError("Can't specify mask together with parc")
        else:
            state['parc'] = parc
            folder = "{parc}"
        self._set_test_options('src', sns_baseline, src_baseline, pmin, tstart,
                               tstop)
        dst = self.get('res-g-deep-file', mkdir=True, fmatch=False, folder=folder,
                       resname="{epoch} {test} {test_options}", ext='html',
                       test=test, **state)
        if not redo and not redo_test and os.path.exists(dst):
            return

        # load data
        ds, res = self.load_test(None, tstart, tstop, pmin, parc, mask, samples,
                                 'src', sns_baseline, src_baseline, True, True,
                                 redo_test)
        y = ds['srcm']

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        self._report_test_info(report.add_section("Test Info"), ds, y, test,
                               tstart, tstop, pmin, res, 'src', include, True)

        model = self._tests[test]['model']
        colors = plot.colors_for_categorial(ds.eval(model))
        legend = None
        if parc is None and pmin in (None, 'tfce'):
            section = report.add_section("P<=.05")
            _report.source_bin_table(section, res, 0.05)
            clusters = res.find_clusters(0.05, maps=True)
            clusters.sort('tstart')
            title = "{tstart}-{tstop} {location} p={p}{mark} {effect}"
            for cluster in clusters.itercases():
                legend = _report.source_time_cluster(section, cluster, y, model,
                                                     ds, title, colors, legend)

            # trend section
            section = report.add_section("Trend: p<=.1")
            _report.source_bin_table(section, res, 0.1)

            # not quite there section
            section = report.add_section("Anything: P<=.2")
            _report.source_bin_table(section, res, 0.2)
        elif parc and pmin in (None, 'tfce'):
            # add picture of parc
            section = report.add_section(parc)
            caption = "Labels in the %s parcellation." % parc
            self._report_parc_image(section, caption)

            # add subsections for individual labels
            title = "{tstart}-{tstop} p={p}{mark} {effect}"
            for label in y.source.parc.cells:
                section = report.add_section(label.capitalize())

                clusters_sig = res.find_clusters(0.05, True, source=label)
                clusters_trend = res.find_clusters(0.1, True, source=label)
                clusters_trend = clusters_trend.sub("p>0.05")
                clusters_all = res.find_clusters(0.2, True, source=label)
                clusters_all = clusters_all.sub("p>0.1")
                clusters = combine((clusters_sig, clusters_trend, clusters_all))
                clusters.sort('tstart')
                src_ = y.sub(source=label)
                legend = _report.source_time_clusters(section, clusters, src_,
                                                      ds, model, include,
                                                      title, colors, legend)
        elif parc is None:  # thresholded, whole brain
            if mask:
                title = "Whole Brain Masked by %s" % mask.capitalize()
                section = report.add_section(title)
                caption = "Mask: %s" % mask.capitalize()
                self._report_parc_image(section, caption)
            else:
                section = report.add_section("Whole Brain")

            _report.source_bin_table(section, res)

            clusters = res.find_clusters(include, maps=True)
            clusters.sort('tstart')
            title = "{tstart}-{tstop} {location} p={p}{mark} {effect}"
            _report.source_time_clusters(section, clusters, y, ds, model,
                                         include, title, colors, legend)
        else:  # thresholded, parc
            # add picture of parc
            section = report.add_section(parc)
            caption = "Labels in the %s parcellation." % parc
            self._report_parc_image(section, caption)
            _report.source_bin_table(section, res)

            # add subsections for individual labels
            title = "{tstart}-{tstop} p={p}{mark} {effect}"
            for label in y.source.parc.cells:
                section = report.add_section(label.capitalize())

                clusters = res.find_clusters(None, True, source=label)
                src_ = y.sub(source=label)
                legend = _report.source_time_clusters(section, clusters, src_,
                                                      ds, model, include,
                                                      title, colors, legend)

        # report signature
        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))

        report.save_html(dst)

    def make_report_rois(self, test, parc=None, pmin=None, tstart=0.15, tstop=None,
                         samples=10000, sns_baseline=True, src_baseline=None,
                         redo=False, **state):
        """Create an HTML report on ROI time courses

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        parc : str
            Parcellation that defines ROIs.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline : None | True | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction (None).
        redo : bool
            If the target file already exists, delete and recreate it.
        """
        parc = self.get('parc', parc=parc)
        if not parc:
            raise ValueError("No parcellation specified")
        self._set_test_options('src', sns_baseline, src_baseline, pmin, tstart,
                               tstop)
        dst = self.get('res-g-deep-file', mkdir=True, fmatch=False,
                       folder="%s ROIs" % parc.capitalize(),
                       resname="{epoch} {test} {test_options}",
                       ext='html', test=test, **state)
        if not redo and os.path.exists(dst):
            return

        # load data
        group = self.get('group')
        ds = self.load_evoked_stc(group, sns_baseline, src_baseline, ind_stc=True)

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # method intro (compose it later when data is available)
        info_section = report.add_section("Test Info")

        # add parc image
        section = report.add_section(parc)
        caption = "ROIs in the %s parcellation." % parc
        self._report_parc_image(section, caption)

        # load labels
        with self._temporary_state:
            labels = self._load_labels(mrisubject=self.get('common_brain'),
                                       match=False)
        labels_lh = []
        labels_rh = []
        for label in labels:
            if label.startswith('unknown'):
                continue
            elif label.endswith('-lh'):
                labels_lh.append(label)
            elif label.endswith('-rh'):
                labels_rh.append(label)
            else:
                raise NotImplementedError("Label named %s" % repr(label.name))
        labels_lh.sort()
        labels_rh.sort()

        # add content body
        model = self._tests[test]['model']
        colors = plot.colors_for_categorial(ds.eval(model))
        label_cache = CacheDict(self._load_labels, 'subject')
        for hemi, label_names in (('Left', labels_lh), ('Right', labels_rh)):
            section = report.add_section("%s Hemisphere" % hemi)
            for label in label_names:
                key = self._add_stc_label(ds, label, label_cache)
                y = ds[key]
                res = self._make_test(y, ds, test, samples, pmin, tstart, tstop,
                                      None, None)
                _report.roi_timecourse(section, ds, label, model, res, colors)

        # compose info
        self._report_test_info(info_section, ds, y, test, tstart, tstop, pmin,
                               res, 'src')

        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst)

    def make_report_eeg(self, test, pmin=None, tstart=0.15, tstop=None,
                        samples=10000, baseline=True, include=1,
                        redo=False, redo_test=False, **state):
        """Create an HTML report on EEG sensor space spatio-temporal clusters

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        baseline : None | True | tuple
            Apply baseline correction using this period. True (default) to use
            the epoch's baseline specification.
        include : 0 < scalar <= 1
            Create plots for all clusters with p-values smaller or equal this
            value (the default is 1, i.e. to show all clusters).
        redo : bool
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        redo_test : bool
            Redo the test even if a cached file exists.
        """
        self._set_test_options('eeg', baseline, None, pmin, tstart, tstop)
        dst = self.get('res-g-deep-file', mkdir=True, fmatch=False,
                       folder="EEG Spatio-Temporal",
                       resname="{epoch} {test} {test_options}",
                       ext='html', test=test, modality='eeg', **state)
        if not redo and not redo_test and os.path.exists(dst):
            return

        # load data
        ds, res = self.load_test(None, tstart, tstop, pmin, None, None, samples,
                                 'sns', baseline, None, True, True, redo_test)
        y = ds['eeg']

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        info_section = report.add_section("Test Info")
        self._report_test_info(info_section, ds, y, test, tstart, tstop, pmin,
                               res, 'sns', include, True)

        # add connectivity image
        p = plot.SensorMap(ds['eeg'], show=False)
        p.show_connectivity()
        image_conn = p.image("connectivity.png")
        info_section.add_figure("Sensor map with connectivity", image_conn)
        p.close()

        model = self._tests[test]['model']
        colors = plot.colors_for_categorial(ds.eval(model))
        legend = None
        if pmin in (None, 'tfce'):
            section = report.add_section("P<=.05")
            _report.sensor_bin_table(section, res, 0.05)
            clusters = res.find_clusters(0.05, maps=True)
            clusters.sort('tstart')
            for cluster in clusters.itercases():
                legend = _report.sensor_time_cluster(section, cluster, y, model,
                                                     ds, colors, legend)

            # trend section
            section = report.add_section("Trend: p<=.1")
            _report.sensor_bin_table(section, res, 0.1)

            # not quite there section
            section = report.add_section("Anything: P<=.2")
            _report.sensor_bin_table(section, res, 0.2)
        else:
            section = report.add_section("Clusters")
            _report.sensor_bin_table(section, res)
            clusters = res.find_clusters(include, maps=True)
            clusters.sort('tstart')
            for cluster in clusters.itercases():
                legend = _report.sensor_time_cluster(section, cluster, y, model,
                                                     ds, colors, legend)

        report.sign(('eelbrain', 'mne', 'scipy', 'numpy'))
        report.save_html(dst)

    def make_report_eeg_sensors(self, test, sensors=('FZ', 'CZ', 'PZ', 'O1', 'O2'),
                                pmin=None, tstart=0.15, tstop=None,
                                samples=10000, baseline=True, redo=False,
                                **state):
        """Create an HTML report on individual EEG sensors

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        sensors : sequence of str
            Names of the sensors which to include.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        baseline : None | True | tuple
            Apply baseline correction using this period. True (default) to use
            the epoch's baseline specification.
        redo : bool
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        """
        self._set_test_options('eeg', baseline, None, pmin, tstart, tstop)
        dst = self.get('res-g-deep-file', mkdir=True, fmatch=False,
                       folder="EEG Sensors",
                       resname="{epoch} {test} {test_options}",
                       ext='html', test=test, modality='eeg', **state)
        if not redo and os.path.exists(dst):
            return

        # load data
        ds = self.load_evoked(self.get('group'), baseline, True)

        # test that sensors are in the data
        eeg = ds['eeg']
        missing = [s for s in sensors if s not in eeg.sensor.names]
        if missing:
            raise ValueError("The following sensors are not in the data: %s" % missing)

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        info_section = report.add_section("Test Info")

        # add sensor map
        p = plot.SensorMap(ds['eeg'], show=False)
        p.mark_sensors(sensors)
        info_section.add_figure("Sensor map", p)
        p.close()

        # main body
        model = self._tests[test]['model']
        caption = "Signal at %s."
        colors = plot.colors_for_categorial(ds.eval(model))
        for sensor in sensors:
            y = eeg.sub(sensor=sensor)
            res = self._make_test(y, ds, test, samples, pmin, tstart, tstop,
                                  None, None)
            _report.timecourse(report, ds, y, model, res, sensor,
                               caption % sensor, colors)

        self._report_test_info(info_section, ds, eeg, test, tstart, tstop, pmin,
                               res, 'sns')
        report.sign(('eelbrain', 'mne', 'scipy', 'numpy'))
        report.save_html(dst)

    def _report_state(self):
        t = self.show_state(hide=['annot', 'epoch-bare',
                                  'epoch-stim', 'ext', 'hemi', 'label',
                                  'subject', 'model', 'mrisubject'])
        return t

    def _report_subject_info(self, ds, model):
        # add subject information to experiment
        s_ds = table.repmeas('n', model, 'subject', ds=ds)
        s_ds2 = self.show_subjects(asds=True)
        s_ds.update(s_ds2[('subject', 'mri')])
        s_table = s_ds.as_table(midrule=True, count=True, caption="All "
                                "subjects included in the analysis with "
                                "trials per condition")
        return s_table

    def _report_test_info(self, section, ds, y, test, tstart, tstop, pmin,
                          res, data, include=None, spatiotemporal=False):
        info = List("Data:")
        info.add_item(self.format('epoch = {epoch} {evoked-kind} ~ {model}'))
        if data == 'src':
            info.add_item(self.format("cov = {cov}"))
            info.add_item(self.format("inv = {inv}"))
        section.append(info)

        # cluster test info
        if spatiotemporal:
            title = "Spatio-Temporal Cluster Permutation Test:"
        else:
            title = "Temporal Cluster Permutation Test:"
        info = List(title)
        test_params = self._tests[test]
        info.add_item("test = %s  (%s)" % (test_params['kind'], test_params['desc']))
        info.add_item("Time interval:  %i - %i ms." % (_report.tstart(tstart, y.time),
                                                       _report.tstop(tstop, y.time)))
        if pmin is None:
            info.add_item("P-values based on maximum value in randomizations")
        elif pmin == 'tfce':
            info.add_item("Threshold-free cluster enhancement (Smith & Nichols, 2009)")
        else:
            info.add_item("Cluster threshold equivalent to p = %s" % pmin)
            # cluster criteria
            criteria = info.add_sublist("Criteria:")
            mintime = self.cluster_criteria.get('mintime', 0)
            criteria.add_item("Minimum duration:  %i ms" % round(mintime * 1000))
            if spatiotemporal:
                if data == 'src':
                    minsource = self.cluster_criteria.get('minsource', 0)
                    criteria.add_item("At least %i contiguous sources." % minsource)
                elif data == 'sns':
                    minsensor = self.cluster_criteria.get('minsensor', 0)
                    criteria.add_item("At least %i contiguous sensors." % minsensor)

            if include is not None:
                info.add_item("Separate plots of all clusters with a p-value < %s"
                              % include)

        # n samples
        if res.samples == -1:
            info.add_item("All possible permutations (%i)" % res.n_samples)
        else:
            info.add_item("%i permutations" % res.samples)

        if data is not None:
            # spatial cluster test (for temporal tests, res is only
            # representative)
            info.add_item(res.info_list())
        section.append(info)

        section.append(self._report_subject_info(ds, test_params['model']))
        section.append(self._report_state())

    def _report_parc_image(self, section, caption):
        "Add picture of the current parcellation"
        self.store_state()
        self.set(mrisubject=self.get('common_brain'))
        brain = self.plot_annot(w=1000)
        self.restore_state()
        image = plot.brain.image(brain, 'parc.png')
        section.add_image_figure(image, caption)

    def make_src(self, redo=False, **kwargs):
        """Make the source space

        Parameters
        ----------
        redo : bool
            Recreate the source space even if the corresponding file already
            exists.
        """
        dst = self.get('src-file', **kwargs)
        subject = self.get('mrisubject')
        common_brain = self.get('common_brain')

        is_scaled = (subject != common_brain) and is_fake_mri(self.get('mri-dir'))

        if is_scaled:
            # make sure the source space exists for the original
            with self._temporary_state:
                self.make_src(mrisubject=common_brain)
                orig = self.get('src-file')

            if not redo and os.path.exists(dst):
                if os.path.getmtime(dst) >= os.path.getmtime(orig):
                    return

            src = self.get('src')
            subjects_dir = self.get('mri-sdir')
            mne.scale_source_space(subject, src, subjects_dir=subjects_dir)
        elif not redo and os.path.exists(dst):
            return
        else:
            src = self.get('src')
            kind, param = src.split('-')
            if kind == 'vol':
                mri = self.get('mri-file')
                bem = self.get('bem-file', fmatch=True)
                mne.setup_volume_source_space(subject, dst, pos=float(param),
                                              mri=mri, bem=bem, mindist=0.,
                                              exclude=0.,
                                              subjects_dir=self.get('mri-sdir'))
            else:
                spacing = kind + param
                mne.setup_source_space(subject, dst, spacing, overwrite=redo,
                                       subjects_dir=self.get('mri-sdir'),
                                       add_dist=True)

    def _make_test(self, y, ds, test, samples, pmin, tstart, tstop, dist_dim,
                   parc_dim):
        """just compute the test result

        Parameters
        ----------
        y : NDVar
            Dependent variable.
        ds : Dataset
            Other variables.
        p : dict
            Test parameters.
        ...
        """
        # find cluster criteria
        kwargs = {'samples': samples, 'tstart': tstart, 'tstop': tstop,
                  'dist_dim': dist_dim, 'parc': parc_dim}
        if pmin == 'tfce':
            kwargs['tfce'] = True
        elif pmin is not None:
            kwargs['pmin'] = pmin
            if 'mintime' in self.cluster_criteria:
                kwargs['mintime'] = self.cluster_criteria['mintime']

            if y.has_dim('source') and 'minsource' in self.cluster_criteria:
                kwargs['minsource'] = self.cluster_criteria['minsource']
            elif y.has_dim('sensor') and 'minsensor' in self.cluster_criteria:
                kwargs['minsensor'] = self.cluster_criteria['minsensor']

        # perform test
        p = self._tests[test]
        kind = p['kind']
        if kind == 'ttest_rel':
            res = testnd.ttest_rel(y, p['model'], p['c1'], p['c0'], 'subject',
                                   ds=ds, tail=p.get('tail', 0), **kwargs)
        elif kind == 't_contrast_rel':
            res = testnd.t_contrast_rel(y, p['model'], p['contrast'], 'subject',
                                        ds=ds, tail=p.get('tail', 0), **kwargs)
        elif kind == 'anova':
            res = testnd.anova(y, p['x'], match='subject', ds=ds, **kwargs)
        else:
            raise RuntimeError("Test kind=%s" % repr(kind))

        return res

    def next(self, field='subject', group=None):
        """Change field to the next value

        Parameters
        ----------
        field : str
            The field for which the value should be changed (default 'subject').
        group : str
            If cycling through subjects, only use subjects form that group.
            Does not set change the experiment's group value.
        """
        current = self.get(field)
        if field == 'subject':
            if group is None:
                values = self._get_group_members(self.get('group'))
            else:
                values = self._get_group_members(group)
        else:
            values = self.get_field_values(field)

        # find the index of the next value
        if current in values:
            idx = values.index(current) + 1
            if idx == len(values):
                idx = -1
        else:
            for idx in xrange(len(values)):
                if values[idx] > current:
                    break
            else:
                idx = -1

        # set the next value
        if idx == -1:
            next_ = values[0]
            print("The last %s was reached; rewinding to "
                  "%r" % (field, next_))
        else:
            next_ = values[idx]
            print("%s: %r -> %r" % (field, current, next_))
        self.set(**{field: next_})

    def plot_annot(self, surf='inflated', views=['lat', 'med'], hemi=None,
                   borders=False, alpha=0.7, w=600, parc=None):
        """Plot the annot file on which the current parcellation is based

        kwargs are for self.plot_brain().

        Parameters
        ----------
        surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
            Freesurfer surface to use as brain geometry.
        views : str | iterator of str
            View or views to show in the figure.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot (default includes hemisphere with more than one
            label in the annot file).
        borders : bool | int
            Show only label borders (PySurfer Brain.add_annotation() argument).
        alpha : scalar
            Alpha of the annotation (1=opaque, 0=transparent, default 0.7).
        w : int
            Figure width per hemisphere.
        parc : None | str
            Parcellation to plot. If None, use parc from the current state.
        """
        if parc is None:
            parc = self.get('parc')
        else:
            parc = self.get('parc', parc=parc)

        self.make_annot()
        mri_sdir = self.get('mri-sdir')
        if is_fake_mri(self.get('mri-dir')):
            subject = self.get('common_brain')
        else:
            subject = self.get('mrisubject')

        brain = plot.brain.annot(parc, subject, surf, borders, alpha, hemi, views,
                                 w, subjects_dir=mri_sdir)
        return brain

    def plot_brain(self, surf='inflated', title=None, hemi='lh', views=['lat'],
                   w=500, clear=True, common_brain=True):
        """Create a PuSyrfer Brain instance

        Parameters
        ----------
        w : int
            Total window width.
        clear : bool
            If self.brain exists, replace it with a new plot (if False,
            the existsing self.brain is returned).
        common_brain : bool
            If the current mrisubject is a scaled MRI, use the common_brain
            instead.
        """
        import surfer
        if clear:
            self.brain = None
        else:
            if self.brain is None:
                pass
            elif self.brain._figures == [[None, None], [None, None]]:
                self.brain = None
            else:
                return self.brain

        # find subject
        mri_sdir = self.get('mri-sdir')
        if common_brain and is_fake_mri(self.get('mri-dir')):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)
        else:
            mrisubject = self.get('mrisubject')

        if title is not None:
            title = title.format(mrisubject=mrisubject)

        if hemi in ('lh', 'rh'):
            self.set(hemi=hemi)
            height = len(views) * w * 3 / 4.
        else:
            height = len(views) * w * 3 / 8.

        config_opts = dict(background=(1, 1, 1), foreground=(0, 0, 0),
                           width=w, height=height)
        brain = surfer.Brain(mrisubject, hemi, surf, True, title, config_opts,
                             None, mri_sdir, views)

        self.brain = brain
        return brain

    def plot_coreg(self, ch_type=None, **kwargs):
        """Plot the coregistration (Head shape and MEG helmet)

        Parameters
        ----------
        ch_type : 'meg' | 'eeg'
            Plot only MEG or only EEG sensors (default is both).
        """
        self.set(**kwargs)
        raw = self.load_raw()
        return mne.viz.plot_trans(raw.info, self.get('trans-file'),
                                  self.get('subject'), self.get('mri-sdir'),
                                  ch_type, 'head')

    def plot_whitened_gfp(self, s_start=None, s_stop=None, run=None):
        """Plot the GFP of the whitened evoked to evaluate the the covariance matrix

        Parameters
        ----------
        s_start : str
            Subject at which to start (default is the first subject).
        s_stop: str
            Subject at which to stop (default is the last subject).
        run : bool
            Run the GUI after plotting (default depends on environment).
        """
        gfps = []
        subjects = []
        with self._temporary_state:
            self.set(model='')
            for subject in self.iter_range(s_start, s_stop):
                cov = self.load_cov()
                picks = range(len(cov.ch_names))
                ds = self.load_evoked(baseline=True)
                whitened_evoked = mne.whiten_evoked(ds[0, 'evoked'], cov, picks)
                gfp = whitened_evoked.data.std(0)

                gfps.append(gfp)
                subjects.append(subject)

        colors = plot.colors_for_oneway(subjects)
        title = "Whitened Global Field Power (%s)" % self.get('cov')
        fig = plot._base.Figure(1, title, h=7, run=run)
        ax = fig._axes[0]
        for subject, gfp in izip(subjects, gfps):
            ax.plot(whitened_evoked.times, gfp, label=subject,
                    color=colors[subject])
        ax.legend(loc='right')
        fig.show()
        return fig

    def plot_evoked(self, subject=None, separate=False, baseline=True, run=None,
                    **kwargs):
        """Plot evoked sensor data

        Parameters
        ----------
        subject : str
            Subject or group name (default is current subject).
        separate : bool
            When plotting a group, plot all subjects separately instead or the group
            average (default False).
        baseline : None | True | tuple
            Apply baseline correction using this period. True (default) to use
            the epoch's baseline specification.
        run : bool
            Run the GUI after plotting (default in accordance with plotting
            default).
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        model = self.get('model') or None
        if subject:
            ds = self.load_evoked(baseline=baseline)
            return plot.TopoButterfly('meg', model, ds=ds, title=subject,
                                      run=run)
        elif separate:
            for subject in self.iter(group=group):
                ds = self.load_evoked(baseline=baseline)
                plot.TopoButterfly('meg', model, ds=ds, title=subject, run=False)

            if run or plot._base.do_autorun():
                gui.run()
        else:
            ds = self.load_evoked(group, baseline=baseline)
            return plot.TopoButterfly('meg', model, ds=ds, title=subject,
                                      run=run)

    def plot_label(self, label, surf='inflated', w=600, clear=False):
        """Plot a label"""
        if isinstance(label, basestring):
            label = self.load_label(label)
        title = label.name

        brain = self.plot_brain(surf, title, 'split', ['lat', 'med'], w, clear)
        brain.add_label(label, alpha=0.75)
        return brain

    def run_mne_analyze(self, subject=None, modal=False):
        subjects_dir = self.get('mri-sdir')
        subject = subject or self.get('mrisubject')
        fif_dir = self.get('raw-dir', subject=subject)
        subp.run_mne_analyze(fif_dir, subject=subject,
                             subjects_dir=subjects_dir, modal=modal)

    def run_mne_browse_raw(self, subject=None, modal=False):
        fif_dir = self.get('raw-dir', subject=subject)
        subp.run_mne_browse_raw(fif_dir, modal)

    def set(self, subject=None, **state):
        """
        Set variable values.

        Parameters
        ----------
        subject : str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        *other* : str
            All other keywords can be used to set templates.
        """
        if subject is not None:
            state['subject'] = subject
            if 'mrisubject' not in state:
                if 'mri' in state:
                    mri = state['mri']
                else:
                    mri = self.get('mri')
                state['mrisubject'] = self._mri_subjects[mri][subject]

        FileTree.set(self, **state)

    def _eval_group(self, group):
        if group not in self.get_field_values('group'):
            if group not in self.get_field_values('subject'):
                raise ValueError("No group or subject named %r" % group)
        return group

    def _post_set_group(self, _, group):
        if group != 'all' and self.get('root'):
            group_members = self._get_group_members(group)
            if self.get('subject') not in group_members:
                self.set(group_members[0])

    def set_inv(self, ori='free', snr=3, method='dSPM', depth=None,
                pick_normal=False):
        """Set the type of inverse solution used for source estimation

        Parameters
        ----------
        ori : 'free' | 'fixed' | float ]0, 1]
            Orientation constraint (default 'free'; use a float to specify a
            loose constraint).
        snr : scalar
            SNR estimate for regularization (default 3).
        method : 'MNE' | 'dSPM' | 'sLORETA'
            Inverse method.
        depth : None | float
            Depth weighting (default None).
        pick_normal : bool
            Pick the normal component of the estimated current vector (default
            False).
        """
        if not isinstance(ori, basestring):
            ori = 'loose%s' % str(ori)[1:]
        items = [ori, str(snr), method]

        if depth:
            items.append(str(depth))

        if pick_normal:
            items.append('pick_normal')

        inv = '-'.join(items)
        self.set(inv=inv)

    @staticmethod
    def _eval_inv(inv):
        m = inv_re.match(inv)
        if m is None:
            raise ValueError("Invalid inverse specification: inv=%r" % inv)

        ori, snr, method, depth, pick_normal = m.groups()
        if ori.startswith('loose'):
            loose = float(ori[5:])
            if not 0 <= loose <= 1:
                err = ('First value of inv (loose parameter) needs to be '
                       'in [0, 1]')
                raise ValueError(err)

        return inv

    def _post_set_inv(self, _, inv):
        if '*' in inv:
            self._params['make_inv_kw'] = None
            self._params['apply_inv_kw'] = None
            return

        m = inv_re.match(inv)
        ori, snr, method, depth, pick_normal = m.groups()

        make_kw = {}
        apply_kw = {}

        if ori == 'fixed':
            make_kw['fixed'] = True
            make_kw['loose'] = None
        elif ori == 'free':
            make_kw['loose'] = 1
        elif ori.startswith('loose'):
            make_kw['loose'] = float(ori[5:])

        if depth is not None:
            make_kw['depth'] = float(depth)

        apply_kw['method'] = method
        apply_kw['lambda2'] = 1. / float(snr) ** 2
        if pick_normal:
            apply_kw['pick_normal'] = True

        self._params['make_inv_kw'] = make_kw
        self._params['apply_inv_kw'] = apply_kw

    def _eval_model(self, model):
        if model == '':
            return model
        elif len(model) > 1 and '*' in model:
            raise ValueError("Specify model with '%' instead of '*'")

        factors = [v.strip() for v in model.split('%')]

        # find order value for each factor
        ordered_factors = {}
        unordered_factors = []
        for factor in sorted(factors):
            assert_is_legal_dataset_key(factor)
            if factor in self._model_order:
                v = self._model_order.index(factor)
                ordered_factors[v] = factor
            else:
                unordered_factors.append(factor)

        # recompose
        model = [ordered_factors[v] for v in sorted(ordered_factors)]
        if unordered_factors:
            model.extend(unordered_factors)
        return '%'.join(model)

    def _eval_parc(self, parc):
        # Freesurfer parcellations
        if parc == '' or parc in FS_PARCS or parc in FSA_PARCS or parc in EELBRAIN_PARCS:
            return parc
        elif self.parcs is None or parc in self.parcs:
            return parc
        else:
            raise ValueError("Unknown parcellation:  parc=%r" % parc)

    def _post_set_rej(self, _, rej):
        if rej == '*':
            self._params['rej'] = None
        else:
            rej_args = self.epoch_rejection[rej]
            self._params['rej'] = rej_args
            self._fields['cov-rej'] = rej_args.get('cov-rej', rej)

    def set_root(self, root, find_subjects=False):
        """Set the root of the file hierarchy

        Parameters
        ----------
        root : str
            Path of the new root.
        find_subjects : bool
            Update the list of available subjects based on the file hierarchy.
        """
        if root is None:
            root = ''
            find_subjects = False
        self.set(root=root)
        if find_subjects:
            self._update_subject_values()

    def _update_subject_values(self):
        subjects = set()
        sub_dir = self.get(self._subject_loc)
        if os.path.exists(sub_dir):
            for dirname in os.listdir(sub_dir):
                isdir = os.path.isdir(os.path.join(sub_dir, dirname))
                if isdir and self._subject_re.match(dirname):
                    subjects.add(dirname)
        else:
            err = ("Subjects directory not found: %r. Initialize with "
                   "root=None or find_subjects=False, or specifiy proper "
                   "directory in experiment._subject_loc." % sub_dir)
            raise IOError(err)

        subjects = sorted(subjects)
        self._field_values['subject'] = subjects

        if len(subjects) == 0:
            print("Warning: no subjects found in %r" % sub_dir)
            return

        if self.get('subject') not in subjects:
            self.set(subject=subjects[0])

    def _post_set_test(self, _, test):
        if test != '*':
            self.set(model=self._tests[test]['model'])

    def _set_test_options(self, data, sns_baseline, src_baseline, pmin, tstart,
                          tstop):
        """Set templates for test paths with test parameters

        Can be set before or after the test template.

        Parameters
        ----------
        data : 'sns' | 'src'
            Whether the analysis is in sensor or source space.
        ...
        src_baseline :
            Should be None if data=='sns'.
        """
        # data kind (sensor or source space)
        if data == 'sns':
            analysis = '{sns-kind} {evoked-kind}'
        elif data == 'src':
            analysis = '{src-kind} {evoked-kind}'
        elif data == 'eeg':
            analysis = '{eeg-kind} {evoked-kind}'
        else:
            raise ValueError("data=%r. Needs to be 'sns', 'src' or 'eeg'" % data)

        # test properties
        items = []

        # baseline
        # default is baseline correcting in sensor space
        epoch_baseline = self.epochs[self.get('epoch')]['baseline']
        if src_baseline is None:
            if sns_baseline is None:
                items.append('nobl')
            elif sns_baseline not in (True, epoch_baseline):
                items.append('bl=%s' % _time_window_str(sns_baseline))
        else:
            if sns_baseline in (True, epoch_baseline):
                items.append('snsbl')
            elif sns_baseline:
                items.append('snsbl=%s' % _time_window_str(sns_baseline))

            if src_baseline in (True, epoch_baseline):
                items.append('srcbl')
            else:
                items.append('srcbl=%s' % _time_window_str(src_baseline))

        # pmin
        if pmin is not None:
            items.append(str(pmin))

        # time window
        if tstart is not None or tstop is not None:
            items.append(_time_window_str((tstart, tstop)))

        self.set(test_options=' '.join(items), analysis=analysis)

    def show_file_status(self, temp, col=None, row='subject', *args, **kwargs):
        """Compile a table about the existence of files

        Parameters
        ----------
        temp : str
            The name of the path template for the files to examine.
        col : None | str
            Field over which to alternate columns (default is a single column).
        row : str
            Field over which to alternate rows (default 'subject').
        count : bool
            Add a column with a number for each line (default True).
        present : str
            String to display when a given file is present (default 'X').
        absent : str
            String to display when a given file is absent (default '-').
        others :
            ``self.iter()`` kwargs.

        Examples
        --------
        >>> e.show_file_status('raw-file')
            Subject   Raw-file
        ----------------------
        0   A0005     -
        1   A0008     X
        2   A0014     X
        3   A0032     X
        >>> e.show_file_status('rej-file', 'raw')
            Subject   Clm   0.16-40   0-40   1-40
        -----------------------------------------
        0   A0005     -     -         -      -
        1   A0008     -     -         -      -
        2   A0014     -     -         -      -
        3   A0032     -     -         -      -
        """
        return FileTree.show_file_status(self, temp, row, col, *args, **kwargs)

    def show_reg_params(self, asds=False, **kwargs):
        """Show the covariance matrix regularization parameters

        Parameters
        ----------
        asds : bool
            Return a dataset with the parameters (default False).
        """
        if kwargs:
            self.set(**kwargs)
        subjects = []
        reg = []
        for subject in self:
            path = self.get('cov-info-file')
            if os.path.exists(path):
                with open(path, 'r') as fid:
                    text = fid.read()
                reg.append(float(text.strip()))
            else:
                reg.append(float('nan'))
            subjects.append(subject)
        ds = Dataset()
        ds['subject'] = Factor(subjects)
        ds['reg'] = Var(reg)
        if asds:
            return ds
        else:
            print ds

    def show_subjects(self, mri=True, mrisubject=False, caption=True,
                      asds=False):
        """Create a Dataset with subject information

        Parameters
        ----------
        mri : bool
            Add a column specifying whether the subject is using a scaled MRI
            or whether it has its own MRI.
        mrisubject : bool
            Add a column showing the MRI subject corresponding to each subject.
        caption : bool | str
            Caption for the table (For True, use the default "Subject in group
            {group}".
        asds : bool
            Return the table as Dataset instead of an FMTxt Table.
        """
        # caption
        if caption is True:
            caption = self.format("Subjects in group {group}")

        subject_list = []
        mri_list = []
        mrisubject_list = []
        for subject in self.iter():
            subject_list.append(subject)
            mrisubject_ = self.get('mrisubject')
            mrisubject_list.append(mrisubject_)
            if mri:
                mri_dir = self.get('mri-dir')
                if not os.path.exists(mri_dir):
                    mri_list.append('*missing')
                elif is_fake_mri(mri_dir):
                    mri_sdir = self.get('mri-sdir')
                    info = mne.coreg.read_mri_cfg(mrisubject_, mri_sdir)
                    cell = "%s * %s" % (info['subject_from'],
                                        str(info['scale']))
                    mri_list.append(cell)
                else:
                    mri_list.append(mrisubject_)

        ds = Dataset(caption=caption)
        ds['subject'] = Factor(subject_list)
        if mri:
            ds['mri'] = Factor(mri_list)
        if mrisubject:
            ds['mrisubject'] = Factor(mrisubject_list)

        if asds:
            return ds
        else:
            return ds.as_table(midrule=True, count=True)

    def show_input_tree(self):
        """Print a tree of the files needed as input

        See Also
        --------
        show_tree: show complete tree (including secondary, optional and cache)
        """
        return self.show_tree(fields=['raw-file', 'trans-file', 'mri-dir'])
