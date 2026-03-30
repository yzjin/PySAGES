import numpy as onp
import pysages
from pysages.colvars import Distance
from pysages.grids import Grid
# from pysages.methods import ABF
from typing import Optional
from jax import numpy as np
from jax.numpy import linalg
from pysages.colvars.core import CollectiveVariable, AxisCV
from pysages.colvars.coordinates import barycenter, weighted_barycenter

def fitted_positions(positions):
#def fitted_positions(positions, weights):
    #if weights is None:
    #    pos_b = barycenter(positions)
    #else:
    #    pos_b = weighted_barycenter(positions, weights)
    pos_b = barycenter(positions)
    fit_pos = np.add(positions, -pos_b)
    return fit_pos


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = linalg.svd(C, full_matrices=False)
    dh = linalg.det(V) * linalg.det(W)
    S = S.at[-1].multiply(np.sign(dh))
    V = V.at[:, -1].multiply(np.sign(dh))
    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


#def rmsd_kabsch(r1: np.ndarray, Q: np.ndarray, w_0: Optional[np.ndarray]):
def rmsd_kabsch(r1: np.ndarray, Q: np.ndarray):
    """
    Calculate the rmsd respect to a reference using quaternions.

    Parameters
    ----------
    r1: np.ndarray
        Atomic positions.
    r2: np.ndarray
        Cartesian coordinates of the reference position of the atoms.
    w_0: np.ndarray
        Weights of the selected atom positions
    """
    #P = fitted_positions(r1, w_0)
    P = fitted_positions(r1)
    U = kabsch(P, Q)
    optimal_P = np.dot(P, U.T)
    error = np.sqrt(np.sum(np.square(optimal_P - Q)) / P.shape[0])
    return error


class RMSD_Kabsch(CollectiveVariable):
    """
    Use a reference to calculate the RMSD of a set of atoms.
    The algorithm is based on https://doi.org/10.1107/S0567739476001873.
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices.
    references: list[tuple(float)]
       Cartesian coordinates of the reference position of the atoms in indices.
       The coordinates must match the ones of the atoms used in indices.
    """

    #def __init__(self, indices, references, weights=None):
    def __init__(self, indices, references):
        #if weights is not None and len(indices) != len(weights):
        #    raise RuntimeError("Indices and weights must be of the same length")
        super().__init__(indices)
        self.references = np.asarray(references)
        #self.weights = np.asarray(weights)
        #self.Q = fitted_positions(self.references, self.weights)
        self.Q = fitted_positions(self.references)

    @property
    def function(self):
        #return lambda r: rmsd_kabsch(r, self.Q, self.weights)
        return lambda r: rmsd_kabsch(r, self.Q)


