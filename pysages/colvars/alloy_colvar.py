from jax import numpy as np, vmap, jit
from pysages.colvars.core import CollectiveVariable
import numpy as onp


# ---------- MIC distance ------------------------------------------------
def dist_mic(r1, r2, cell):
    frac = (r1 - r2) @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return np.linalg.norm(frac @ cell)


# All pair MIC distance
v_dist_mic = jit(vmap(vmap(dist_mic, (None, 0, None)), (0, None, None)))


# ---------- n = 6, m = 12 switching kernel ------------------------------
@jit
def switch_kernel(r, r0, eps=1.0e-12):
    s = (r / r0) ** 9
    return (1.0 - s) / (1.0 - s ** (14 / 9) + eps)


v_kernel = jit(vmap(vmap(switch_kernel, (None, None, None)), (0, None, None)))


# ---------- coordination number -----------------------------------------
def coord_number(group_A, group_B, cell, r0):
    dists = v_dist_mic(group_A, group_B, cell)  # |A| × |B|
    weights = switch_kernel(dists, r0)  # broadcasting
    return np.sum(weights)


coord_number = jit(coord_number, static_argnums=3)


# ---------- main CV class -----------------------------------------------
def ethane_coord(r, Cidx, Hidx, cell, r0_f, r0_b, alpha, gamma, k):
    pos_C = r[Cidx]
    pos_H = r[Hidx]
    dists = v_dist_mic(pos_C, pos_H, cell)

    c1 = switch_kernel(dists[0, :], r0_f).sum()
    c2 = switch_kernel(dists[1, :], r0_f).sum()
    h1 = switch_kernel(dists[0, :], r0_b).sum()
    h2 = switch_kernel(dists[1, :], r0_b).sum()

    w = 1.0 / (1.0 + np.exp(k * (h1 - h2)))  # soft selector
    P = c1 + c2
    C_CH3 = (1 - w) * c1 + w * c2
    D = abs(c1 - c2)

    penalty = 0.5 * gamma * ((D - 1) + abs(D - 1))  # ⟨D – 1⟩
    S = P - alpha * (3 - C_CH3) + penalty
    return S


class EthaneDehydroCV(CollectiveVariable):
    def __init__(
        self,
        indices,
        cell=None,
        r0_f=1.40,
        r0_b=1.80,
        alpha=1.5,
        gamma=10.0,
        k=7.0,
    ):
        super().__init__(indices)
        self.Cidx = np.array([0, 1])
        self.Hidx = np.array([2, 3, 4, 5, 6, 7])
        self.cell = cell
        self.r0_f, self.r0_b = r0_f, r0_b
        self.alpha, self.gamma, self.k = alpha, gamma, k

    # ------------------------------------------------------------------
    @property
    def function(self):
        return lambda r: ethane_coord(
            r,
            self.Cidx,
            self.Hidx,
            self.cell,
            self.r0_f,
            self.r0_b,
            self.alpha,
            self.gamma,
            self.k,
        )


def switch(dist, r0, num, den):
    return (1 - (dist / r0) ** num) / (1 - (dist / r0) ** den)


def coordnum(r, r0, num, den, cell):
    group1 = r[:2]
    group2 = r[2:]
    dists = v_dist_mic(group1, group2, cell)
    num = switch(dists, r0, num, den)
    return num.sum()


class Coordnum(CollectiveVariable):
    def __init__(self, indices, r0=None, num=None, den=None, cell=None):
        super().__init__(indices)  # first two indices must be carbon
        self.r0 = r0
        self.num = num
        self.den = den
        self.cell = cell

    @property
    def function(self):
        return lambda r: coordnum(r, self.r0, self.num, self.den, self.cell)

