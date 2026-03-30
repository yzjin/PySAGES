# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Whitelist Proton Transfer CV (O–H whitelist + “other O” penalty).

This CV reproduces your validated Python/Numpy version:
  CV = gate * xi - kappa_off * (1 - gate)
with gate = A * U, where
  - xi is a smooth progress variable using ONLY the two endpoints
    (O_init and O_final) via soft O–H occupancies,
  - A promotes configurations where H binds one of the whitelisted O's,
  - U penalizes H being too close to any non-whitelisted O (“other O's”).

Indices layout (important): positions are taken in THIS order
    [H] + allowed_O (ordered as you pass them) + other_O
so the function can slice them deterministically.
"""

import jax.numpy as np
from pysages.colvars.core import CollectiveVariable

_EPS = 1.0e-12


def _norm(v, axis=-1):
    return np.sqrt(np.sum(v * v, axis=axis) + _EPS)


def _soft_occupancy(d, d0, m):
    # omega(d) = 1 / (1 + (d/d0)^m)
    x = (d / d0) ** m
    return 1.0 / (1.0 + x)


def _allowed_gate(S_allowed, sigmaA, nA):
    # A = 1 - exp[-(S/sigmaA)^nA], clipped to [0,1]
    x = (S_allowed / sigmaA) ** nA
    A = 1.0 - np.exp(-x)
    return np.clip(A, 0.0, 1.0)


def _other_gate(dmin, r0, delta, beta, alpha):
    # Smooth step around r0: s = 1 / (1 + exp(-beta*(dmin - r0)/delta))
    # Then sharpen with power alpha: U = s^alpha
    s = 1.0 / (1.0 + np.exp(-beta * (dmin - r0) / (delta + _EPS)))
    U = s ** alpha
    return np.clip(U, 0.0, 1.0)


class WhitelistProtonCV(CollectiveVariable):
    """
    Parameters
    ----------
    h : int
        Index of the hydrogen.
    o_init : int
        Index of initial oxygen (binding site).
    o_final : int
        Index of final oxygen (binding site).
    allowed : list[int]
        Whitelisted O indices (MUST include o_init and o_final).
    others : list[int]
        All *other* oxygen indices (used for the environmental penalty).
    d0, m : float
        Soft O–H occupancy parameters (default d0=1.20 Å, m=6).
    sigmaA, nA : float
        Allowed-gate parameters (default 0.40, 4).
    r0_other, delta_other, beta_other, alpha_other : float
        “Other O” gate parameters (defaults 1.35 Å, 0.10 Å, 12, 6).
    kappa_off : float
        Offset for disallowed states; CV ∈ [-kappa_off, 1] (default 0.60).
    xi_power : float
        Optional sharpening of progress xi (default 1.0 = off).
    """

    def __init__(
        self,
        h,
        o_init,
        o_final,
        allowed,
        others,
        d0=1.20,
        m=6.0,
        sigmaA=0.40,
        nA=4.0,
        r0_other=1.35,
        delta_other=0.10,
        beta_other=12.0,
        alpha_other=6.0,
        kappa_off=0.60,
        xi_power=1.0,
    ):
        # indices order: [H] + allowed + others
        indices = [h] + list(allowed) + list(others)
        super().__init__(indices, group_length=len(indices))

        self._K = len(allowed)
        self._M = len(others)

        # map o_init / o_final into allowed-slice local indices
        try:
            self._i_init = int(allowed.index(o_init))
            self._i_final = int(allowed.index(o_final))
        except ValueError as exc:
            raise RuntimeError("o_init and o_final must be elements of 'allowed'.") from exc

        # params
        self.d0 = float(d0)
        self.m = float(m)
        self.sigmaA = float(sigmaA)
        self.nA = float(nA)
        self.r0_other = float(r0_other)
        self.delta_other = float(delta_other)
        self.beta_other = float(beta_other)
        self.alpha_other = float(alpha_other)
        self.kappa_off = float(kappa_off)
        self.xi_power = float(xi_power)

    @property
    def function(self):
        K = self._K
        i_init = self._i_init
        i_final = self._i_final

        d0 = self.d0
        m = self.m
        sigmaA = self.sigmaA
        nA = self.nA
        r0 = self.r0_other
        delta = self.delta_other
        beta = self.beta_other
        alpha = self.alpha_other
        kappa_off = self.kappa_off
        xi_pow = self.xi_power

        def _cv(positions):
            # positions: [H] + allowed[K] + others[M]
            pH = positions[0]
            p_allowed = positions[1 : 1 + K]
            p_others = positions[1 + K :]

            # H–O distances
            d_allowed = _norm(p_allowed - pH, axis=1)
            omega = _soft_occupancy(d_allowed, d0, m)  # shape (K,)

            # progress using ONLY endpoints
            w_init = omega[i_init]
            w_final = omega[i_final]
            xi = w_final / (w_init + w_final + _EPS)
            if xi_pow != 1.0:
                xi = xi ** xi_pow

            # allowed gate A
            S_allowed = np.sum(omega)  # occupancy mass on whitelist
            A = _allowed_gate(S_allowed, sigmaA, nA)

            # “other O” gate U: use min distance to non-whitelisted O
            d_other = _norm(p_others - pH, axis=1) if p_others.size else np.array([1e6])
            dmin_other = np.min(d_other)
            U = _other_gate(dmin_other, r0, delta, beta, alpha)

            gate = A * U

            # final scalar CV
            cv = gate * xi - kappa_off * (1.0 - gate)
            return cv  # scalar (shape ())

        return _cv