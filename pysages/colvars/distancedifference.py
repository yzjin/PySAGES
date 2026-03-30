# SPDX-License-Identifier: MIT
# hydrogen_transfer.py
"""
Collective variable for proton transfer in Zr-UIO-66:
    CV  =  d(O1-H) − d(O2-H)

• CV < 0 → H closer to O1
• CV > 0 → H closer to O2
• CV ≈ 0 → H midway between O1 and O2
"""

from jax import numpy as np
from jax.numpy import linalg

from pysages.colvars.core import ThreePointCV


# ---------------------------------------------------------------------
#  Low-level function (mirrors the style of `angle`, `dihedral_angle`…)
# ---------------------------------------------------------------------
def distance_difference(p1, p2, p3):
    """
    Parameters
    ----------
    p1, p2, p3 : jax.Array
        3-D coordinates passed by PySAGES (barycentres of the
        user-supplied atom groups).

    Returns
    -------
    float
        d(O1-H) − d(O2-H)
    """
    d1 = linalg.norm(p1 - p3)     # O1–H
    d2 = linalg.norm(p2 - p3)     # O2–H
    return d1 - d2


# ---------------------------------------------------------------------
#  CV class (mirrors the `Angle` class you posted)
# ---------------------------------------------------------------------
class DistanceDifference(ThreePointCV):
    """
    Difference of the two O–H distances.

    Parameters
    ----------
    indices : list[int] | list[tuple[int]]
        **Exactly three** atom (or atom-group) selections, in order:
        1. O1
        2. O2
        3. H  (the transferring proton)

        Example
        -------
        >>> cv = DistanceDifference(indices=[[15], [32], [47]])
    """

    @property
    def function(self):
        return distance_difference