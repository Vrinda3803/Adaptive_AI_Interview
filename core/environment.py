from dataclasses import dataclass, field
import numpy as np

from .knowledge_graph import REG
from .bayesian_irt import (
    GRID,
    DIFF_B,
    DIFF_NAMES,
    TYPE_NAMES,
    ITEM_A,
    ITEM_G,
    ITEM_S,
    COST,
    ND,
    NT,
    KMAX,
    A_MAX,
    NF,
    NG,
    sig,
    bin_ent,
    item_prob_table,
    Belief,
)

ARCHETYPES = {"novice": -1.0, "intermediate": 0.0, "expert": 1.0, "specialist": 0.0, "impostor": -0.4}
RESUME_MU = {-1: -0.4, 0: -0.8, 1: 0.0, 2: 0.9}
RESUME_SD = {-1: 1.3, 0: 1.1, 1: 1.1, 2: 1.1}
HELDOUT_DOMAINS = ("Cybersecurity", "Mobile Development", "Specialized Technologies")
SEEN_DOMAINS = tuple(d for d in REG.domains if d not in HELDOUT_DOMAINS)


@dataclass
class EnvCfg:
    allowed_domains: tuple = None
    archetype_probs: dict = field(
        default_factory=lambda: {"novice": 0.25, "intermediate": 0.30, "expert": 0.20, "specialist": 0.25}
    )
    eps_model: float = 0.05
    eps_true: float = 0.05
    a_scale_true: float = 1.0
    use_graph: bool = True
    use_eig_feat: bool = True
    use_sigma_feat: bool = True
    use_mask: bool = True
    use_resume_prior: bool = True
    use_adaptive_stopping: bool = True
    stop_uncertainty_threshold: float = 0.42
    min_questions_before_stop: int = 1
    r_info: float = 1.0
    r_coverage: float = 0.20
    r_cost: float = 0.03
    r_err: float = 2.0
    r_dec: float = 0.5
    r_invalid: float = 0.3
    r_conf_bonus: float = 0.15
    graph_rho: float = 0.5
    max_per_skill: int = 4
    k_range: tuple = (4, 8)
    extra_budget: tuple = (3, 7)


def make_candidate(rng, reg, probs):
    names = list(probs)
    pr = np.array([probs[k] for k in names], float)
    arche = names[rng.choice(len(names), p=pr / pr.sum())]
    D = len(reg.domains)
    dom = rng.normal(0, 0.5, D)
    if arche == "specialist":
        strong = rng.choice(D, 2, replace=False)
        dom -= 0.4
        dom[strong] += 1.8
    theta = ARCHETYPES[arche] + rng.normal(0, 0.3) + dom[reg.dom_idx] + rng.normal(0, 0.8, reg.N)
    for v in reg.topo:
        for u in reg.prereq[v]:
            theta[v] = 0.75 * theta[v] + 0.25 * theta[u]
    theta = np.clip(theta, -3, 3)
    noisy = theta + (1.0 if arche == "impostor" else 0.0) + rng.normal(0, 1.1, reg.N)
    claimed = rng.random(reg.N) < sig(1.5 * (noisy + 0.3)) if arche != "impostor" else rng.random(reg.N) < 0.85
    level = np.where(noisy < -0.5, 0, np.where(noisy < 0.6, 1, 2))
    claim = np.where(claimed, level, -1)
    return arche, theta, claim


class InterviewEnv:
    """
    Observation = dict(g=global[NG], f=action-features[A_MAX,NF], m=valid-mask[A_MAX]).
    Action index = ((slot * 3) + difficulty) * 2 + type.
    """

    def __init__(self, cfg=None, reg=REG):
        self.cfg = cfg or EnvCfg()
        self.reg = reg
        c = self.cfg
        self.P1 = item_prob_table(1.0, c.eps_model)
        self.HP1 = bin_ent(self.P1)
        self._seed_counter = 0
        self.sim = True

    def reset(self, seed=None):
        c, reg = self.cfg, self.reg
        if seed is None:
            seed = self._seed_counter
            self._seed_counter += 1
        rng = np.random.default_rng(seed)
        # Keep response noise reproducible and paired to the episode seed.
        self.rng = rng
        self.sim = True
        doms = list(c.allowed_domains) if c.allowed_domains else list(reg.domains)
        allowed = np.array([i for i in range(reg.N) if reg.domain[i] in doms])
        K = int(min(rng.integers(c.k_range[0], c.k_range[1] + 1), len(allowed)))
        pd_ = rng.choice(doms, size=min(2, len(doms)), replace=False)
        cand = np.array([i for i in allowed if reg.domain[i] in pd_])
        rest = np.setdiff1d(allowed, cand)
        pool = np.concatenate([rng.permutation(cand), rng.permutation(rest)])
        role = pool[:K]
        w = rng.dirichlet(np.full(K, 2.0))
        base = rng.choice([-0.3, 0.0, 0.5])
        req = base + rng.normal(0, 0.2, K)
        B = int(min(K + rng.integers(c.extra_budget[0], c.extra_budget[1] + 1), 16))
        arche, theta, claim = make_candidate(rng, reg, c.archetype_probs)
        self.arche, self.theta = arche, theta
        self._setup(role, w, req, B, claim)
        return self._obs()

    def reset_custom(self, role_ids, weights, req, claim, budget):
        """Real candidate interview: competency beliefs update via incoming observations."""
        self.sim = False
        self.theta = None
        self.arche = "real"
        w = np.asarray(weights, float)
        w = w / max(w.sum(), 1e-6)
        self._setup(np.asarray(role_ids), w, np.asarray(req, float), int(budget), np.asarray(claim))
        return self._obs()

    def _setup(self, role, w, req, B, claim):
        c, reg = self.cfg, self.reg
        self.role, self.w, self.req, self.B, self.K = np.asarray(role), w, req, B, len(role)
        self.claim = claim
        if c.use_resume_prior:
            mu0 = np.array([RESUME_MU.get(int(claim[k]), -0.4) for k in range(reg.N)])
            sd0 = np.array([RESUME_SD.get(int(claim[k]), 1.3) for k in range(reg.N)])
        else:
            mu0, sd0 = np.zeros(reg.N), np.full(reg.N, 1.5)
        self.belief = Belief(mu0, sd0)
        self.prior_mu = mu0.copy()
        self.asked = np.zeros((KMAX, ND, NT), bool)
        self.cnt = np.zeros(KMAX, int)
        self.t = 0
        self.n_invalid = 0
        self.ep_return = 0.0
        self.log = []
        self.done = False
        self.stopped_early = False

    def decode(self, a):
        slot = a // 6
        return slot, int(self.role[slot]), (a % 6) // 2, a % 2

    def label(self, a):
        s, sk, d, ty = self.decode(a)
        return f"{self.reg.names[sk]} | {DIFF_NAMES[d]} | {TYPE_NAMES[ty]}"

    def _wH(self):
        return float((self.w * self.belief.ent[self.role]).sum())

    def _valid(self):
        K = self.K
        v = np.zeros((KMAX, ND, NT), bool)
        ok = ~self.asked[:K] & (self.cnt[:K] < self.cfg.max_per_skill)[:, None, None]
        ty0 = np.arange(NT)[None, None, :] == 0
        ok = ok & (self.reg.coding_ok[self.role][:, None, None] | ty0)
        v[:K] = ok
        return v.reshape(-1)

    def _obs(self):
        c, reg, b, R, K = self.cfg, self.reg, self.belief, self.role, self.K
        mu, sd = b.mu, b.sd
        mu_r, sd_r = mu[R], sd[R]
        w = (self.w * K)[:, None, None]
        P = b.p[R]
        pm = np.einsum("kg,dtg->kdt", P, self.P1)
        eig = bin_ent(pm) - np.einsum("kg,dtg->kdt", P, self.HP1)

        Fm = np.zeros((K, ND, NT, NF))
        bd = DIFF_B[None, :, None]
        mr = mu_r[:, None, None]
        Fm[..., 0] = w
        Fm[..., 1] = mr
        Fm[..., 2] = sd_r[:, None, None]
        Fm[..., 3] = bd
        Fm[..., 4] = bd - mr
        Fm[..., 5] = np.abs(bd - mr)
        Fm[..., 6] = np.arange(NT)[None, None, :]
        Fm[..., 7] = eig * 3
        Fm[..., 8] = eig * 3 * w
        Fm[..., 9] = (self.cnt[:K] / c.max_per_skill)[:, None, None]
        Fm[..., 10] = (self.cnt[:K] > 0)[:, None, None]
        nb = reg.W.sum(1) + 1e-6
        Fm[..., 11] = ((reg.W @ mu) / nb)[R][:, None, None]
        Fm[..., 12] = ((reg.W @ sd) / nb)[R][:, None, None]
        Fm[..., 13] = (reg.Ppre @ mu)[R][:, None, None]
        Fm[..., 14] = COST[None, None, :] / 2.5
        Fm[..., 15] = (np.clip(np.abs(mu_r - self.req) / sd_r, 0, 4) / 2)[:, None, None]
        Fm[..., 16] = (self.req - mu_r)[:, None, None]
        Fm[..., 17] = self.asked[:K]
        Fm[..., 18] = reg.coding_ok[R][:, None, None]

        if not c.use_graph:
            Fm[..., [11, 12, 13]] = 0
        if not c.use_eig_feat:
            Fm[..., [7, 8]] = 0
        if not c.use_sigma_feat:
            Fm[..., [2, 12, 15]] = 0

        f = np.zeros((A_MAX, NF), np.float32)
        f[: K * 6] = Fm.reshape(K * 6, NF)
        valid = self._valid()
        struct = np.zeros(A_MAX, bool)
        struct[: K * 6] = True
        self._valid_mask = valid
        m = valid if c.use_mask else struct

        g = np.array(
            [
                self.t / 16,
                (self.B - self.t) / 16,
                float((self.cnt[:K] > 0).mean()),
                float((self.w * sd_r).sum()),
                float((self.w * np.abs(mu_r - self.req)).sum()),
                K / 8,
                self.B / 16,
                float((sd_r < 0.5).mean()),
            ],
            np.float32,
        )
        return dict(g=g, f=f, m=m.copy())

    def p_true(self, th, d, ty):
        c = self.cfg
        pc = ITEM_G[ty] + (1 - ITEM_G[ty] - ITEM_S[ty]) * sig(ITEM_A[ty] * c.a_scale_true * (th - DIFF_B[d]))
        return pc * (1 - c.eps_true) + (1 - pc) * c.eps_true

    def step(self, a, score=None, evaluator_confidence=1.0):
        c = self.cfg
        assert not self.done, "Episode already finished"
        self.t += 1
        info = {"invalid": 0, "stopped_early": False}
        slot = a // 6

        if slot >= self.K or not self._valid_mask[a]:
            self.n_invalid += 1
            info["invalid"] = 1
            r = -c.r_invalid
            self.ep_return += r
            if self.t >= self.B:
                self.done = True
            return self._obs(), r, self.done, info

        sk = int(self.role[slot])
        d = (a % 6) // 2
        ty = a % 2

        self.asked[slot, d, ty] = True
        self.cnt[slot] += 1
        wH_before = self._wH()

        # Score resolution
        if score is None:
            score = 1.0 if self.rng.random() < self.p_true(self.theta[sk], d, ty) else 0.0
        score = float(score)

        p1 = self.P1[d, ty]
        # Bayesian update with evaluator reliability calibration
        self.belief.update(
            sk,
            score,
            p1,
            W=self.reg.W if c.use_graph else None,
            rho=c.graph_rho,
            evaluator_confidence=evaluator_confidence,
        )

        dH = wH_before - self._wH()
        cov = float((self.cnt[: self.K] > 0).mean())
        r = c.r_info * dH + c.r_coverage * cov - c.r_cost * COST[ty]

        # Confidence reward bonus when skill uncertainty drops below threshold
        if self.belief.sd[sk] < c.stop_uncertainty_threshold:
            r += c.r_conf_bonus

        # Check adaptive stopping condition
        max_role_sd = float(self.belief.sd[self.role].max())
        all_skills_covered = cov >= 0.99
        min_questions = max(self.K, int(c.min_questions_before_stop))
        confident_enough = max_role_sd <= c.stop_uncertainty_threshold and all_skills_covered and self.t >= min_questions

        if c.use_adaptive_stopping and confident_enough:
            self.done = True
            self.stopped_early = True
            info["stopped_early"] = True

        if self.t >= self.B or not self._valid().any():
            self.done = True

        if self.done and self.sim and self.theta is not None:
            mu_r = self.belief.mu[self.role]
            th_r = self.theta[self.role]
            wrmse = float(np.sqrt((self.w * (mu_r - th_r) ** 2).sum()))
            dec_acc = float((self.w * ((mu_r >= self.req) == (th_r >= self.req))).sum())
            succ = int(wrmse <= 0.50 and dec_acc >= 0.85 and cov >= 0.99)
            r += -c.r_err * wrmse + c.r_dec * dec_acc
            info["final"] = dict(
                wrmse=wrmse,
                dec_acc=dec_acc,
                success=succ,
                coverage=cov,
                stopped_early=self.stopped_early,
                questions_asked=self.t,
            )

        self.ep_return += r
        return self._obs(), r, self.done, info

    def close(self):
        pass
