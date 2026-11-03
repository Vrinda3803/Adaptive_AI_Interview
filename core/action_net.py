import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayesian_irt import NG, NF, A_MAX


class ActionNet(nn.Module):
    """
    Permutation-equivariant scorer: every (skill, difficulty, type) is described by FEATURES
    (not hard-coded skill IDs), so the policy transfers to unseen skills.
    Returns per-action scores and a state value.
    """

    def __init__(self, H=96):
        super().__init__()
        self.gctx = nn.Sequential(nn.Linear(NG, H), nn.ReLU())
        self.aenc = nn.Sequential(nn.Linear(NF, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, 1))
        self.vhead = nn.Sequential(nn.Linear(2 * H, H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, g, f, m):
        c = self.gctx(g)
        e = self.aenc(f)
        mf = m.unsqueeze(-1).float()
        pool = (e * mf).sum(1) / mf.sum(1).clamp(min=1)
        x = torch.cat([e, c.unsqueeze(1).expand_as(e), pool.unsqueeze(1).expand_as(e)], -1)
        return self.head(x).squeeze(-1), self.vhead(torch.cat([c, pool], -1)).squeeze(-1)


class LinearQ(nn.Module):
    """Q-learning with linear function approximation: Q(s,a) = w . [phi(a), phi(a)*remaining_budget]."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2 * NF, 1)

    def forward(self, g, f, m):
        return self.lin(torch.cat([f, f * g[:, None, 1:2]], -1)).squeeze(-1), None


def q_values(net, g, f, m, dueling=False):
    adv, v = net(g, f, m)
    if dueling:
        mf = m.float()
        adv = v[:, None] + adv - (adv * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1)
    return adv


def to_t(obs):
    return (
        torch.from_numpy(obs["g"])[None].float(),
        torch.from_numpy(obs["f"])[None].float(),
        torch.from_numpy(obs["m"])[None].bool(),
    )


class Scorer:
    """Greedy deployable policy wrapping a trained network."""

    def __init__(self, net, kind="ppo", name="PPO"):
        self.net = net
        self.kind = kind
        self.name = name

    @torch.no_grad()
    def __call__(self, obs, env=None):
        g, f, m = to_t(obs)
        if self.kind != "ppo":
            s = q_values(self.net, g, f, m, dueling=(self.kind == "dueling"))
        else:
            s = self.net(g, f, m)[0]
        return int(s.masked_fill(~m, -1e9).argmax(1))

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


def load_policy(path):
    """Load a policy checkpoint saved from the research experiments."""
    ck = torch.load(path, map_location="cpu")
    net = LinearQ() if ck.get("algo") == "QLearning-Linear" else ActionNet()
    net.load_state_dict(ck["state"])
    return Scorer(net, ck.get("kind", "ppo"), ck.get("algo", "PPO"))
