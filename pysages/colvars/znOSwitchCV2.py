# SPDX-License-Identifier: MIT
# Two CVs: (1) progress xi (Zn switches O1->O2), (2) contact mass S=w1+w2

import jax.numpy as jnp
from pysages.colvars.core import CollectiveVariable

_EPS = 1.0e-12

def _norm(v): return jnp.sqrt(jnp.sum(v * v) + _EPS)

def _w_cn(d, r0=2.2, n=6.0, m=12.0):
    x = d / r0
    num = 1.0 - x**n
    den = 1.0 - x**m
    w = num / (den + _EPS)
    w = jnp.where(jnp.abs(x - 1.0) < 1e-6, n / m, w)   # limit at r0
    return jnp.clip(w, 0.0, 1.0)

def _sigmoid(z): return 1.0 / (1.0 + jnp.exp(-z))

class ZnO12SwitchCV(CollectiveVariable):
    """Progress xi \in (0,1): Zn switches O1 -> O2."""
    def __init__(self, zn, o1, o2, r0=2.2, n=6.0, m=12.0, ksig=0.15):
        super().__init__([zn, o1, o2], group_length=3)
        self.r0, self.n, self.m, self.ksig = float(r0), float(n), float(m), float(ksig)

    @property
    def function(self):
        r0, n, m, ksig = self.r0, self.n, self.m, self.ksig
        def _cv(positions):
            pZn, pO1, pO2 = positions[0], positions[1], positions[2]
            d1 = _norm(pZn - pO1); d2 = _norm(pZn - pO2)
            w1 = _w_cn(d1, r0, n, m); w2 = _w_cn(d2, r0, n, m)
            xi = _sigmoid((w2 - w1) / (ksig + _EPS))
            return xi
        return _cv

class ZnOContactMassCV(CollectiveVariable):
    """Contact mass S = w1 + w2 \in [0,1]."""
    def __init__(self, zn, o1, o2, r0=2.2, n=6.0, m=12.0):
        super().__init__([zn, o1, o2], group_length=3)
        self.r0, self.n, self.m = float(r0), float(n), float(m)

    @property
    def function(self):
        r0, n, m = self.r0, self.n, self.m
        def _cv(positions):
            pZn, pO1, pO2 = positions[0], positions[1], positions[2]
            d1 = _norm(pZn - pO1); d2 = _norm(pZn - pO2)
            w1 = _w_cn(d1, r0, n, m); w2 = _w_cn(d2, r0, n, m)
            S = jnp.clip(w1 + w2, 0.0, 1.0)
            return S
        return _cv

