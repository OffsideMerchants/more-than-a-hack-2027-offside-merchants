"""Edge-aware dense GATv2 + team readout + dilated TCN, and a pooled (no-message-passing) twin."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NMAX = 27  # present nodes kept per frame (22 players + duplicates/fragments + ball)


def compact(nodes, pres):
    """Move present nodes first and keep NMAX. nodes [B,T,N,F] pres [B,T,N]."""
    order = torch.argsort((~pres).to(torch.int8), dim=-1, stable=True)[..., :NMAX]
    nodes = torch.gather(nodes, 2, order[..., None].expand(-1, -1, -1, nodes.shape[-1]))
    return nodes, torch.gather(pres, 2, order)


def edge_feats(nodes):
    """nodes [M,N,8] -> [M,N,N,8]: dist, rel vel(2), closing speed, bearing(cos,sin), same-team, ball-edge."""
    pos, vel = nodes[..., 0:2] * 52.5, nodes[..., 2:4] * 8
    dp = pos[:, None, :, :] - pos[:, :, None, :]           # j - i
    dv = vel[:, None, :, :] - vel[:, :, None, :]
    dist = dp.norm(dim=-1, keepdim=True)
    u = dp / dist.clamp_min(1e-3)
    closing = -(dv * u).sum(-1, keepdim=True)
    team = nodes[..., 5:7]
    same = (team[:, :, None, :] * team[:, None, :, :]).sum(-1, keepdim=True)
    ball = 1 - team.sum(-1)
    be = (ball[:, :, None] + ball[:, None, :]).unsqueeze(-1)
    return torch.cat([dist / 30, dv / 8, closing / 8, u, same, be], -1)


class GATv2(nn.Module):
    def __init__(self, h, heads=4, ed=8):
        super().__init__()
        self.h, self.k = h, heads
        self.wl, self.wr, self.we = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(ed, h, bias=False)
        self.att = nn.Parameter(torch.randn(heads, h // heads) * 0.1)
        self.out, self.norm = nn.Linear(h, h), nn.LayerNorm(h)

    def forward(self, x, e, adj):
        M, N, H = x.shape
        a = F.leaky_relu(self.wl(x)[:, :, None] + self.wr(x)[:, None] + self.we(e), 0.2)  # [M,i,j,H]
        s = (a.view(M, N, N, self.k, -1) * self.att).sum(-1)                               # [M,i,j,k]
        s = s.masked_fill(~adj[..., None], -1e4)
        w = torch.softmax(s, dim=2)
        v = self.wr(x).view(M, N, self.k, -1)
        o = torch.einsum("mijk,mjkc->mikc", w, v).reshape(M, N, H)
        return self.norm(x + F.gelu(self.out(o)))


def knn_adj(nodes, pres, k):
    pos = nodes[..., :2] * 52.5
    d = torch.cdist(pos, pos).masked_fill(~pres[:, None, :], 1e4).masked_fill(~pres[:, :, None], 1e4)
    eye = torch.eye(d.shape[1], dtype=torch.bool, device=d.device)
    d = d.masked_fill(eye, 1e4)
    kth = d.topk(min(k, d.shape[1] - 1), dim=-1, largest=False).values[..., -1:]
    a = (d <= kth) | (d.transpose(1, 2) <= kth.transpose(1, 2))
    return (a & pres[:, None, :] & pres[:, :, None]) | (eye & pres[:, :, None])


class TCN(nn.Module):
    def __init__(self, c, dil=(1, 2, 4, 8, 16), drop=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d), nn.GELU(),
                                                   nn.Dropout(drop), nn.Conv1d(c, c, 1)) for d in dil])
        self.norms = nn.ModuleList([nn.GroupNorm(1, c) for _ in dil])

    def forward(self, x):  # [B,C,T]
        for b, n in zip(self.blocks, self.norms):
            x = n(x + b(x))
        return x


class PossessionNet(nn.Module):
    """kind='gnn': GATv2 message passing. kind='pool': node MLPs only (same readout, same TCN)."""

    def __init__(self, kind="gnn", h=64, layers=2, k=8, tcn_c=128):
        super().__init__()
        self.kind, self.k = kind, k
        self.enc = nn.Sequential(nn.Linear(8, h), nn.GELU(), nn.Linear(h, h))
        if kind == "gnn":
            self.layers = nn.ModuleList([GATv2(h) for _ in range(layers)])
        else:  # per-node residual MLPs; the width is raised below to match the GNN parameter count
            self.layers = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h), nn.LayerNorm(h))
                                         for _ in range(layers)])
        self.proj = nn.Linear(6 * h + 3, tcn_c)   # [home mean,max | away mean,max | all mean,max | counts]
        self.tcn = TCN(tcn_c)
        self.head = nn.Conv1d(tcn_c, 3, 1)         # alive logit, home score, away score

    @staticmethod
    def readout(z, pres, team):
        outs = []
        for m in (team[..., 0], team[..., 1], (team[..., 0] + team[..., 1]).clamp(max=1)):
            mk = (m > 0.5) & pres
            c = mk.sum(-1, keepdim=True)
            mean = (z * mk[..., None]).sum(-2) / c.clamp_min(1)
            mx = z.masked_fill(~mk[..., None], -1e4).max(-2).values
            mx = torch.where(c > 0, mx, torch.zeros_like(mx))
            outs += [mean, mx]
        cnt = torch.stack([((team[..., 0] > .5) & pres).sum(-1) / 11., ((team[..., 1] > .5) & pres).sum(-1) / 11.,
                           ((team.sum(-1) < .5) & pres).sum(-1).float()], -1)
        return torch.cat(outs + [cnt], -1)

    def forward(self, nodes, pres):
        B, T, N, Fd = nodes.shape
        nodes, pres = compact(nodes, pres)
        N = nodes.shape[2]
        x, p = nodes.reshape(B * T, N, Fd), pres.reshape(B * T, N)
        z = self.enc(x)
        if self.kind == "gnn":
            adj = knn_adj(x, p, self.k)
            e = edge_feats(x)
            for l in self.layers:
                z = l(z, e, adj)
        else:
            for l in self.layers:
                z = z + l(z)
        g = self.readout(z, p, x[..., 5:7]).view(B, T, -1)
        y = self.head(self.tcn(self.proj(g).transpose(1, 2)))  # [B,3,T]
        return y[:, 0], y[:, 1] - y[:, 2]                     # alive logit, home-minus-away logit
