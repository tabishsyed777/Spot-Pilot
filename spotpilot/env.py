"""Two-worker cloud scheduling simulator with integer-credit admission control."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .scenario import generate

# Configuration changes stop affected workers; lost work is restored from checkpoint.
CONFIGS = ((), ('spot',), ('spot','spot'), ('on',), ('on','on'), ('spot','on'))
ACTION_NAMES = ('idle', 'one_spot', 'two_spot', 'one_on_demand', 'two_on_demand', 'mixed', 'checkpoint')
FEATURES = ('time', 'budget_fraction', 'spot_price', 'risk', 'ready_fraction',
            'future_fraction', 'completed_fraction', 'missed_fraction', 'min_slack',
            'ready_remaining', 'active_fraction', 'spot_fraction', 'worker0_progress',
            'worker0_saved', 'worker0_remaining', 'worker0_slack', 'worker1_progress',
            'worker1_saved', 'worker1_remaining', 'worker1_slack', 'recent_loss',
            'budget_per_remaining_work', 'workload_remaining', 'market_phase')


class SpotEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(self, suite='standard', render_mode=None, record=True):
        self.suite, self.render_mode, self.record = suite, render_mode, record
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(0, 1, (len(FEATURES),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario_seed = int(self.np_random.integers(2**31)) if seed is None else int(seed)
        self.scenario = generate(scenario_seed, self.suite)
        self.jobs = self.scenario.jobs
        n = len(self.jobs)
        self.progress, self.saved = [0]*n, [0]*n
        self.completed, self.missed = set(), set()
        self.workers = [None, None]
        self.t = self.spent = self.lost = self.interruptions = self.rejections = 0
        self.recent_loss = 0
        self.done = False
        self.ledger = []
        return self._obs(), {}

    def _ready(self):
        return [j for j in self.jobs if j.arrival <= self.t < j.deadline and j.id not in self.completed | self.missed]

    def _obs(self):
        s = self.scenario
        ready = self._ready()
        remaining = sum(j.work-self.progress[j.id] for j in self.jobs if j.id not in self.completed | self.missed)
        min_slack = min((j.deadline-self.t-(j.work-self.progress[j.id]) for j in ready), default=s.horizon)
        active = [w for w in self.workers if w is not None]
        market = min(self.t,s.horizon-1)
        values = [self.t/s.horizon, (s.budget-self.spent)/s.budget, s.prices[market]/6,
                  s.risks[market], len(ready)/len(self.jobs),
                  sum(j.arrival>self.t for j in self.jobs)/len(self.jobs),
                  len(self.completed)/len(self.jobs), len(self.missed)/len(self.jobs),
                  max(min_slack,0)/s.horizon,
                  sum(j.work-self.progress[j.id] for j in ready)/96,
                  len(active)/2, sum(w['kind']=='spot' for w in active)/2]
        for w in self.workers:
            if w is None: values += [0,0,0,0]
            else:
                j=self.jobs[w['job']]
                values += [self.progress[j.id]/j.work, self.saved[j.id]/j.work,
                           (j.work-self.progress[j.id])/8, (j.deadline-self.t)/s.horizon]
        values += [min(self.recent_loss/16,1), min((s.budget-self.spent)/max(6*remaining,1),1), remaining/96, (self.t%8)/8]
        return np.clip(np.array(values,dtype=np.float32),0,1)

    def step(self, action):
        if self.done: raise RuntimeError('reset required after episode end')
        if not self.action_space.contains(action): raise ValueError('action must be 0..6')
        action = int(action)
        s = self.scenario
        events = []
        before_lost = self.lost
        checkpoint = action == 6

        def release(slot, reason):
            worker = self.workers[slot]
            if worker is not None:
                jid = worker['job']
                loss = self.progress[jid]-self.saved[jid]
                self.progress[jid] = self.saved[jid]
                self.lost += loss
                events.append(dict(type='release',slot=slot,job=jid,reason=reason,lost=loss))
                self.workers[slot] = None

        if not checkpoint:
            desired = CONFIGS[action]
            for slot,w in enumerate(self.workers):
                if w is not None and (slot >= len(desired) or w['kind'] != desired[slot]):
                    release(slot,'resize')
            assigned = {w['job'] for w in self.workers if w is not None}
            ready = sorted((j for j in self._ready() if j.id not in assigned),key=lambda j:(j.deadline,j.id))
            for slot,kind in enumerate(desired):
                if self.workers[slot] is None and ready:
                    j=ready.pop(0)
                    self.workers[slot] = dict(job=j.id,kind=kind)
                    events.append(dict(type='assign',slot=slot,job=j.id,kind=kind))

        cost = sum((s.prices[self.t] if w['kind']=='spot' else s.on_demand_price)
                   + (s.checkpoint_storage_price if checkpoint else 0)
                   for w in self.workers if w is not None)
        rejected = cost > s.budget-self.spent
        if rejected:
            self.rejections += 1
            events.append(dict(type='budget_rejection',requested_cost=cost))
            for slot in range(2): release(slot,'budget')
            cost = 0
        self.spent += cost
        completions = 0
        for slot,w in enumerate(self.workers):
            if w is None: continue
            jid = w['job']
            interrupted = w['kind']=='spot' and s.draws[self.t][slot] < s.risks[self.t]
            events.append(dict(type='execute',slot=slot,job=jid,kind=w['kind'],mode='checkpoint' if checkpoint else 'run',interrupted=interrupted))
            if interrupted:
                self.interruptions += 1
                release(slot,'interruption')
            elif checkpoint:
                self.saved[jid] = self.progress[jid]
                events.append(dict(type='save',slot=slot,job=jid,progress=self.progress[jid]))
            else:
                self.progress[jid] += 1
                events.append(dict(type='work',slot=slot,job=jid,progress=self.progress[jid]))
                if self.progress[jid] == self.jobs[jid].work:
                    self.completed.add(jid); completions += 1
                    events.append(dict(type='complete',slot=slot,job=jid,time=self.t+1))
                    self.workers[slot] = None
        expired = [j for j in self.jobs if j.deadline <= self.t+1 and j.id not in self.completed | self.missed]
        for j in expired:
            for slot,w in enumerate(self.workers):
                if w is not None and w['job']==j.id: release(slot,'deadline')
            self.missed.add(j.id)
            events.append(dict(type='expire',job=j.id))
        self.recent_loss = self.lost-before_lost
        reward = 4*completions - 2*len(expired) - .02*cost - .08*self.recent_loss
        if self.record:
            self.ledger.append(dict(t=self.t,action=action,cost=cost,spent=self.spent,reward=reward,events=events))
        self.t += 1
        terminated = len(self.completed | self.missed)==len(self.jobs)
        truncated = self.t >= s.horizon and not terminated
        self.done = terminated or truncated
        info = dict(completed=len(self.completed),missed=len(self.missed),spent=self.spent,
                    lost_work=self.lost,interruptions=self.interruptions,budget_rejections=self.rejections,
                    completion_rate=len(self.completed)/len(self.jobs))
        return self._obs(),float(reward),terminated,truncated,info

    def render(self):
        return f't={self.t} completed={len(self.completed)}/{len(self.jobs)} spent={self.spent}/{self.scenario.budget} lost_work={self.lost}'
