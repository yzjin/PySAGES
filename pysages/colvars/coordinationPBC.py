from jax import jit, device_put, lax, numpy as np, vmap
from math import exp, pi, sin, sqrt, cos, acos
from pysages.colvars.core import CollectiveVariable, TwoPointCV

class CoordinationPBC(TwoPointCV):

    def __init__(self, indices, kernel, cell):
        super().__init__(indices)
        self.kernel = kernel
        self.cell = cell

    @property
    def function(self):
        """
        Returns
        -------

        Callable
            See `pysages.colvars.pairwise.coordination` for details.
        """
        return jit(lambda p1, p2: cd(p1, p2, self.kernel, self.cell))


def cd(p1, p2, kernel, cell):
    group_length = p1.shape[0]
    vkernel = vmap(vmap(kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))
    return vkernel(p1, p2, cell).sum() #/ group_length


def dist_mic(r1, r2, cell):
    """Calculate minimum image convention (MIC) distance in non-cubic cell."""
    cell_inv = np.linalg.inv(cell)
    d = r1 - r2
    d_scaled = d @ cell_inv
    d_scaled -= np.round(d_scaled)
    d_mic = d_scaled @ cell
    return np.linalg.norm(d_mic)


def kernel(r1, r2, cell, r0=1.6):
    s = (dist_mic(r1, r2, cell) / r0) ** 9
    return (1 - s) / (1 - s ** (14 / 9))
