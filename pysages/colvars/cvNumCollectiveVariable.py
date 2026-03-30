from jax import jit, numpy as np, vmap
from pysages.colvars.core import CollectiveVariable, TwoPointCV

class CvNumCollectiveVariable(TwoPointCV):
    def __init__(self, indices, cell):
        super().__init__(indices)
        self.cell = cell

    @property
    def function(self):
        return jit(lambda p1, p2: compute_cv_num(p1, p2, self.cell))


def dist_mic(r1, r2, cell):
    """Calculate minimum image convention (MIC) distance in non-cubic cell."""
    cell_inv = np.linalg.inv(cell)
    d = r1 - r2
    d_scaled = d @ cell_inv
    d_scaled -= np.round(d_scaled)
    d_mic = d_scaled @ cell
    return np.linalg.norm(d_mic)


def compute_ni(p1, p2, cell, R0):
    def compute_ni_single(p1):
        def compute_term(p2):
            Rij = dist_mic(p1, p2, cell)
            term_num = 1 - (Rij / R0) ** 16
            term_den = 1 - (Rij / R0) ** 56
            return term_num / term_den

        ni = np.sum(vmap(compute_term)(p2))
        return ni

    ni_values = vmap(compute_ni_single)(p1)
    return ni_values


def compute_cv_num(p1, p2, cell):
    R0 = 1.32  # Please define the value of R0 here.
    ni_values = compute_ni(p1, p2, cell, R0)
    cv_num = np.sum((ni_values - 2) ** 2)
    return cv_num
