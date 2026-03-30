from jax import jit, numpy as np, vmap, grad
from pysages.colvars.core import CollectiveVariable, TwoPointCV
import itertools

class CvDistCollectiveVariable(TwoPointCV):

    def __init__(self, indices, cell):
        super().__init__(indices)
        self.cell = cell

    @property
    def function(self):
        return jit(lambda p1, p2: compute_cv_dist(p1, p2, self.cell))

def dist_mic(r1, r2, cell):
    """Calculate minimum image convention (MIC) distance in non-cubic cell."""
    cell_inv = np.linalg.inv(cell)
    d = r1 - r2
    d_scaled = d @ cell_inv
    d_scaled -= np.round(d_scaled)
    d_mic = d_scaled @ cell
    return np.linalg.norm(d_mic)


def compute_qi(p1, p2, cell):  # Assuming p1 is Oxygen and p2 is Hydrogen
    λ = 8  # in A^(-1)

    def compute_qi_single(oxygen_position):
        def compute_term(hydrogen_position):
            Rij = dist_mic(oxygen_position, hydrogen_position, cell)
            sum_term = np.sum(np.array([np.exp(-λ * dist_mic(h, hydrogen_position, cell)) for h in p1]))  # Changed from p2 to p1
            return np.exp(-λ * Rij) / sum_term

        qi = np.sum(vmap(compute_term)(p2))
        return qi - 2

    qi_values = vmap(compute_qi_single)(p1)
    return qi_values


def compute_cv_dist(p1, p2, cell):
    qi_values = compute_qi(p1, p2, cell)
    #oxygen_positions = positions[indices_oxygen]

    def compute_term(i, k):
        Ri = p1[i]
        Rk = p1[k]
        Rik = dist_mic(Ri, Rk, cell)
        term = qi_values[i] * qi_values[k] * Rik
        return term

    indices = np.array(list(itertools.combinations(range(len(p1)), 2)))
    terms = vmap(compute_term, in_axes=(0, 0))(indices[:, 0], indices[:, 1])
    cv_dist = np.sum(terms)

    return cv_dist * (-1)
