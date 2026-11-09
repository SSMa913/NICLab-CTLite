"""Continuous-Time Lightweight forecasting (CTLite).

encode: causal, mask-normalized temporal moments and observation reliability.
decode: bounded local readout and reliability-gated low-rank variable correction.
Times use the normalized coordinates stored in the benchmark splits.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class CTLite(nn.Module):

    def __init__(self, channels, rank, width, rates):
        nn.Module.__init__(self)
        self.n = channels
        if not 1 <= rank <= self.n or width < 1:
            raise ValueError('Rank must be in [1, channels]; width must be positive')
        if len(rates) < 2 or rates[0] != 0 or any((not math.isfinite(x) or x < 0 for x in rates)):
            raise ValueError('Keep at least two finite nonnegative scales, including zero')
        if any((a >= b for a, b in zip(rates, rates[1:]))):
            raise ValueError('Decay rates must be strictly increasing')
        self.register_buffer('rates', torch.tensor(rates, dtype=torch.float32))
        self.k = len(rates)
        self.weights = nn.Parameter(torch.zeros(self.n, 2 * self.k + 2))
        self.bias = nn.Parameter(torch.zeros(self.n))
        self.log_damping = nn.Parameter(torch.full((self.n,), 1.0))
        self.borrow_in = nn.Parameter(torch.randn(self.n, rank) / math.sqrt(self.n))
        self.borrow_out = nn.Parameter(torch.zeros(rank, self.n))
        self.gate = nn.Parameter(torch.zeros(self.n, 3))
        self.shared = nn.Sequential(nn.Linear(2 * self.k + 5, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.shared[-1].weight)
        nn.init.zeros_(self.shared[-1].bias)

    def encode(self, x, x_mark, x_mask):
        """Return multiscale levels/slopes and channel reliability from history."""
        m = (x_mask > 0).to(x.dtype)
        x = torch.where(m.bool(), x, torch.zeros_like(x))
        t = x_mark[..., 0] if x_mark.ndim == 3 else x_mark
        # Center times at the latest observed history timestamp across channels.
        observed = m.any(-1)
        origin = t.masked_fill(~observed, -1000000.0).amax(1).clamp_min(0)
        tr = t - origin[:, None]
        # The lower exponent bound is numerical protection, not a time horizon.
        w = torch.exp((self.rates[None, :, None] * tr[:, None]).clamp(-60, 0))
        mass = torch.bmm(w, m).clamp_min(1e-06)
        means = torch.bmm(w, x) / mass
        mt = torch.bmm(w * tr[:, None], m) / mass
        mtt = torch.bmm(w * tr[:, None].square(), m) / mass
        mtx = torch.bmm(w * tr[:, None], x) / mass
        variance = (mtt - mt.square()).clamp_min(0)
        # Ridge stabilization and effective-count shrinkage limit unsupported slopes.
        slope = (mtx - mt * means) / (variance + 0.001)
        neff = mass.square() / torch.bmm(w.square(), m).clamp_min(1e-06)
        slope = slope * ((neff - 1).clamp_min(0) / (neff + 1))
        last_t, idx = t[:, :, None].expand_as(m).masked_fill(~m.bool(), -1000000.0).max(1)
        valid = m.any(1).to(x.dtype)
        last = x.gather(1, idx[:, None]).squeeze(1) * valid
        age = (origin[:, None] - last_t).clamp(0, 1) * valid + (1 - valid)
        count = m.sum(1)
        return dict(last=last, means=means.transpose(1, 2), slope=slope.transpose(1, 2), count=count, age=age, valid=valid, origin=origin)

    def decode(self, z, y_mark):
        """Predict at query timestamps using bounded local and cross-variable terms."""
        q = y_mark[..., 0] if y_mark.ndim == 3 else y_mark
        dt = (q - z['origin'][:, None]).clamp_min(0)
        damp = F.softplus(self.log_damping)[None, None, :]
        elapsed = dt[:, :, None] + z['age'][:, None, :]
        horizon = -torch.expm1(-elapsed * damp) / damp
        last = z['last'][:, None, :]
        means = z['means'][:, None, :, :].expand(-1, q.shape[1], -1, -1)
        delta = means - last[:, :, :, None]
        slope = z['slope'][:, None, :, :] * horizon[:, :, :, None]
        features = torch.cat([delta, slope, last.expand(-1, q.shape[1], -1)[..., None], (last * torch.expm1(-elapsed * damp))[..., None]], -1)
        pred = last + (features * self.weights[None, None]).sum(-1) + self.bias
        reliability = torch.stack([z['age'], 1 / (z['count'] + 1), z['valid']], -1)
        shared_inputs = torch.cat([features, reliability[:, None].expand(-1, q.shape[1], -1, -1)], -1)
        pred = pred + self.shared(shared_inputs).squeeze(-1)
        confidence = z['valid'] * torch.exp(-z['age'] * F.softplus(self.log_damping)[None])
        context = z['last'] * confidence @ self.borrow_in @ self.borrow_out
        gate = torch.sigmoid((reliability * self.gate[None]).sum(-1))
        pred = pred + context[:, None, :] * gate[:, None, :] * torch.exp(-dt[:, :, None] * damp)
        return pred

    def forward(self, x, x_mark, x_mask, y_mark, **unused):
        """Predict [batch, query time, channel]; targets never enter the model."""
        return self.decode(self.encode(x, x_mark, x_mask), y_mark)
