from jax import jit, device_put, lax, numpy as np, vmap
import jax.debug
from math import exp, pi, sin, sqrt, cos, acos
from pysages.colvars.core import CollectiveVariable, TwoPointCV

class CoordinationPBC(TwoPointCV):

    def __init__(self, indices, kernel, cell, r0=1.6):
        super().__init__(indices)
        self.kernel = kernel
        self.cell = cell
        self.r0 = r0

    @property
    def function(self):
        """
        Returns
        -------

        Callable
            See `pysages.colvars.pairwise.coordination` for details.
        """
        return jit(lambda p1, p2: cd(p1, p2, self.kernel, self.cell, r0=self.r0))


def cd(p1, p2, kernel, cell, r0):
    vkernel = vmap(vmap(kernel, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
    return vkernel(p1, p2, cell, r0).sum() #/ group_length


def cd_sqdiff(p1, p2, kernel, cell, r0, d):
    vkernel = vmap(vmap(kernel, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
    diff = vkernel(p1, p2, cell, r0).sum(axis=1) - d
    sq = np.sum(np.square(diff))
    return sq


class CombinedCoordinationPBC(CollectiveVariable):
    def __init__(self, indices, kernel, cell, d, r0=1.6):
        super().__init__(indices)
        self.kernel = kernel
        self.cell = cell
        self.r0 = r0
        self.d = d

    @property
    def function(self):
        """
        Returns
        -------

        Callable
            See `pysages.colvars.pairwise.coordination` for details.
        """
        return jit(lambda p1, p2: cd_sqdiff(
            p1, p2, self.kernel, self.cell, self.r0, self.d))


def cd_sum(p1, p2, kernel, cell, r0, d):
    vkernel = vmap(vmap(kernel, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
    diff = vkernel(p1, p2, cell, r0).sum(axis=1) - d
    sq = np.sum(diff)
    return sq


class CombinedCoordinationPBCDiff(CollectiveVariable):
    def __init__(self, indices, kernel, cell, d=0, r0=1.6):
        super().__init__(indices)
        self.kernel = kernel
        self.cell = cell
        self.r0 = r0
        self.d = d

    @property
    def function(self):
        """
        Returns
        -------

        Callable
            See `pysages.colvars.pairwise.coordination` for details.
        """
        return jit(lambda p1, p2: cd_sum(
            p1, p2, self.kernel, self.cell, self.r0, self.d))


class CoordinationPBCMax(TwoPointCV):
    """
    Coordination CV with PBC, but only counts the maximum value for each item.
    """

    def __init__(self, indices, kernel, cell, r0):
        super().__init__(indices)
        self.kernel = kernel
        self.cell = cell
        self.r0 = r0

    @property
    def function(self):
        """
        Returns
        -------

        Callable
            See `pysages.colvars.pairwise.coordination` for details.
        """
        return jit(lambda p1, p2: cd_max(p1, p2, self.kernel, self.cell, r0=self.r0))


def cd_max(p1, p2, kernel, cell, r0):
    """Calculate coordination number with maximum value for each item in p2."""
    group_length = p1.shape[0]
    vkernel = vmap(vmap(kernel, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))
    return vkernel(p1, p2, cell, r0).max(axis=1).sum()


