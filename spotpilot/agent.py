"""NumPy Double DQN: neural Q function, replay, target network and Adam."""
from pathlib import Path
import numpy as np

class DQN:
    def __init__(self, seed=0, hidden=64):
        self.rng = np.random.default_rng(seed)
        self.p = [self.rng.normal(0, np.sqrt(2/24), (24, hidden)), np.zeros(hidden),
                  self.rng.normal(0, 0.05, (hidden, 7)), np.zeros(7)]
        self.target = [x.copy() for x in self.p]
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
        self.updates = 0

    def values(self, x, target=False):
        w, b, v, c = self.target if target else self.p
        return np.maximum(0, np.asarray(x) @ w+b) @ v+c

    def act(self, obs, epsilon=0):
        return int(self.rng.integers(7)) if self.rng.random() < epsilon else int(self.values(obs).argmax())

    def learn(self, batch, gamma=0.98, lr=0.0007):
        s, a, r, ns, done = batch
        next_actions = self.values(ns).argmax(axis=1)
        y = r + gamma*(1-done)*self.values(ns, target=True)[np.arange(len(s)), next_actions]
        w, b, v, c = self.p
        h = np.maximum(0, s@w+b)
        q = h@v+c
        err = q[np.arange(len(s)), a]-y
        dq = np.zeros_like(q)
        dq[np.arange(len(s)), a] = np.clip(err, -1, 1)/len(s)
        dh = (dq@v.T)*(h>0)
        grads = [s.T@dh, dh.sum(axis=0), h.T@dq, dq.sum(axis=0)]
        self.updates += 1
        for p, m, vv, g in zip(self.p, self.m, self.v, grads):
            m *= .9; m += .1*g
            vv *= .999; vv += .001*g*g
            p -= lr*(m/(1-.9**self.updates))/(np.sqrt(vv/(1-.999**self.updates))+1e-8)
        if self.updates % 250 == 0:
            self.target = [x.copy() for x in self.p]
        return float(np.mean(np.where(abs(err)<1, .5*err**2, abs(err)-.5)))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"p{i}":p for i,p in enumerate(self.p)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            if not all(f"p{i}" in f for i in range(4)):
                raise ValueError("checkpoint is missing network arrays")
            params = [f[f"p{i}"].copy() for i in range(4)]
            if params[0].ndim != 2:
                raise ValueError("checkpoint has an invalid input matrix")
            hidden = params[0].shape[1]
            shapes = [(24, hidden), (hidden,), (hidden, 7), (7,)]
            if hidden < 1 or [p.shape for p in params] != shapes or any(not np.issubdtype(p.dtype, np.number) or not np.isfinite(p).all() for p in params):
                raise ValueError("checkpoint has incompatible shapes or non-finite weights")
            obj = cls(hidden=hidden)
            obj.p = params
            obj.target = [p.copy() for p in obj.p]
        return obj

class Replay:
    def __init__(self, capacity=30000):
        self.s = np.zeros((capacity,24)); self.ns = self.s.copy()
        self.a = np.zeros(capacity,dtype=int); self.r = np.zeros(capacity); self.d = np.zeros(capacity)
        self.pos = self.size = 0
        self.capacity = capacity

    def add(self,s,a,r,ns,d):
        i=self.pos
        self.s[i],self.a[i],self.r[i],self.ns[i],self.d[i]=s,a,r,ns,d
        self.pos=(i+1)%self.capacity
        self.size=min(self.size+1,self.capacity)

    def sample(self,rng,n=64):
        ix=rng.integers(self.size,size=n)
        return self.s[ix],self.a[ix],self.r[ix],self.ns[ix],self.d[ix]
