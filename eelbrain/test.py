"""Some basic statistical tests for univariate data.

.. autosummary::
   :toctree: generated

   anova
   pairwise
   ttest
   correlations
   lilliefors

"""
__test__ = False

from ._stats.glm import anova, ANOVA
from ._stats.test import (pairwise, ttest, correlations, bootstrap_pairwise,
                          lilliefors)
