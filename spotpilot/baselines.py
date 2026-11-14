"""Baselines use only the same observation vector supplied to the learner."""
def idle(obs): return 0

def spot_only(obs): return 2

def on_demand_only(obs): return 4

def deadline_aware(obs):
    # Prefer affordable spot capacity, upgrade for tight slack or high current risk.
    if obs[4] == 0: return 0
    if obs[21] > .65 and (obs[8] < 4/48 or obs[3] >= .12): return 4
    if obs[21] > .4 and (obs[8] < 6/48 or obs[3] >= .12): return 5
    return 2

def checkpoint_aware(obs):
    # Preserve useful spot progress when risk makes a restart costly.
    at_risk = (obs[12]-obs[13])*8 + (obs[16]-obs[17])*8
    if obs[11] > 0 and at_risk >= 3 and obs[3] >= .12 and obs[8] > 3/48:
        return 6
    return deadline_aware(obs)
