import numpy as np
from .env import SpotEnv
from .verifier import verify


def episode(policy, seed, suite='standard', audit=True):
    with SpotEnv(suite=suite,record=audit) as env:
        obs,_=env.reset(seed=seed); total=0
        while True:
            obs,r,term,trunc,info=env.step(int(policy(obs))); total+=r
            if term or trunc: break
        verdict=verify(env.scenario,env.ledger) if audit else None
        if audit and not verdict['success']: raise RuntimeError(f'audit failed: {verdict}')
        result=dict(seed=seed,return_=total,**info)
        if audit: result['audit_success']=verdict['success']
        evidence=dict(scenario=env.scenario.to_dict(),ledger=env.ledger,verdict=verdict)
        return result,evidence


def evaluate(policy, episodes=20, seed=50000, suite='standard', audit=True):
    records=[episode(policy,seed+i,suite,audit)[0] for i in range(episodes)]
    summary={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='seed'}
    return dict(summary=summary,episodes=records)
