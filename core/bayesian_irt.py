import numpy as np
from scipy.special import logsumexp

GRID = np.linspace(-3.5, 3.5, 71)  #latent competency grid
DIFF_B = np.array([-1.2, 0.0, 1.2])  #easy, medium, hard
DIFF_NAMES = ["easy", "medium", "hard"]
TYPE_NAMES = ["conceptual", "coding"]
ITEM_A = np.array([1.7, 2.4])  #discrimination (conceptual, coding)
ITEM_G = np.array([0.15, 0.02])  #guessing
ITEM_S = np.array([0.05, 0.08])  #slip
COST = np.array([1.0, 2.5])  #relative candidate time cost
ND, NT, KMAX = 3, 2, 8
A_MAX, NF, NG = KMAX * ND * NT, 19, 8

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def bin_ent(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def item_prob_table(a_scale=1.0, eps=0.05):
    P = np.zeros((ND, NT, len(GRID)))
    for d in range(ND):
        for t in range(NT):
            pc = ITEM_G[t] + (1 - ITEM_G[t] - ITEM_S[t]) * sig(ITEM_A[t] * a_scale * (GRID - DIFF_B[d]))
            P[d, t] = pc * (1 - eps) + (1 - pc) * eps
    return P

class Belief:
    def __init__(self, mu0, sd0):
        mu0 = np.asarray(mu0, dtype=float)
        sd0 = np.asarray(sd0, dtype=float)
        lp = -0.5 * ((GRID[None, :] - mu0[:, None]) / sd0[:, None]) ** 2
        self.logp = lp - logsumexp(lp, axis=1, keepdims=True)
        self._refresh()

    def _refresh(self):
        self.p = np.exp(self.logp)
        self.mu = self.p @ GRID
        self.sd = np.sqrt(np.maximum(self.p @ (GRID ** 2) - (self.mu ** 2), 1e-6))
        self.ent = -(self.p * self.logp).sum(1)

    def update(self, i, score, p1, W=None, rho=0.0, evaluator_confidence=1.0):
        score = float(np.clip(score, 0.0, 1.0))
        c_eval = float(np.clip(evaluator_confidence, 0.1, 1.0))
        ll = score * np.log(p1 + 1e-12) + (1 - score) * np.log(1 - p1 + 1e-12)
        w = np.zeros(self.logp.shape[0])
        w[i] = 1.0 * c_eval
        if W is not None and rho > 0:
            w = w + rho * c_eval * W[i]
        self.logp = self.logp + w[:, None] * ll[None, :]
        self.logp -= logsumexp(self.logp, axis=1, keepdims=True)
        self._refresh()
