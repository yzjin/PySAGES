from jax import jit, vmap, numpy as np
from pysages.colvars.core import FourPointCV

class CoordinationCombination(FourPointCV):
    """
    Computes a linear combination of two coordination numbers derived from four groups of atoms.
    """

    def __init__(self, indices, kernel1, kernel2, cell, coeff1, coeff2):
        """
        Parameters
        ----------
        indices : list of 4 tuples
            Each tuple contains the indices for one of the four atom groups.
        kernel1 : Callable
            Kernel function for the first coordination number.
        kernel2 : Callable
            Kernel function for the second coordination number.
        cell : jax.Array
            Simulation cell matrix for periodic boundary conditions.
        coeff1 : float
            Coefficient for the first coordination number.
        coeff2 : float
            Coefficient for the second coordination number.
        """
        super().__init__(indices)
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.cell = cell
        self.coeff1 = coeff1
        self.coeff2 = coeff2

    @property
    def function(self):
        """
        Returns
        -------
        Callable
            Function that calculates the combined coordination number.
        """
        return jit(lambda p1, p2, p3, p4: combine_coordination_numbers(
            p1, p2, p3, p4, self.kernel1, self.kernel2, self.cell, self.coeff1, self.coeff2
        ))

def combine_coordination_numbers(p1, p2, p3, p4, kernel1, kernel2, cell, coeff1, coeff2):
    """
    Calculate the linear combination of two coordination numbers.

    Parameters
    ----------
    p1, p2, p3, p4 : jax.Array
        Positions of the four atom groups.
    kernel1, kernel2 : Callable
        Kernel functions for coordination calculations.
    cell : jax.Array
        Simulation cell matrix for periodic boundary conditions.
    coeff1, coeff2 : float
        Coefficients for the linear combination.

    Returns
    -------
    float
        Linear combination of the two coordination numbers.
    """
    # Calculate the first coordination number using kernel1
    coord_num1 = compute_coordination(p1, p2, kernel1, cell)

    # Calculate the second coordination number using kernel2
    coord_num2 = compute_coordination(p3, p4, kernel2, cell)

    # Return the linear combination
    return coeff1 * coord_num1 + coeff2 * coord_num2

def compute_coordination(group1, group2, kernel, cell):
    """
    Compute the coordination number between two groups of atoms.

    Parameters
    ----------
    group1, group2 : jax.Array
        Positions of the two atom groups.
    kernel : Callable
        Kernel function for coordination calculation.
    cell : jax.Array
        Simulation cell matrix for periodic boundary conditions.

    Returns
    -------
    float
        Coordination number.
    """
    vkernel = vmap(vmap(kernel, in_axes=(0, None, None)), in_axes=(None, 0, None))
    return vkernel(group1, group2, cell).sum()