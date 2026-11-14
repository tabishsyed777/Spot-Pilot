import copy
from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from spotpilot.env import SpotEnv
from spotpilot.scenario import Job,generate
from spotpilot.verifier import verify
from spotpilot.evaluation import episode
from spotpilot.baselines import spot_only,on_demand_only,checkpoint_aware
from spotpilot.agent import DQN,Replay


def controlled(horizon=8,budget=100,work=4,deadline=None):
    env=SpotEnv();env.reset(seed=7)
    env.scenario=replace(env.scenario,horizon=horizon,budget=budget,jobs=(Job(0,0,work,deadline or horizon),),prices=(2,)*horizon,risks=(0.,)*horizon,draws=((1.,1.),)*horizon)
    env.jobs=env.scenario.jobs;env.progress=[0];env.saved=[0]
    return env


def finish(env,action=2):
    while not env.done:env.step(action)
    result=verify(env.scenario,env.ledger)
    assert result['success'],result
    return result


def test_gym_contract():
    with SpotEnv() as env:check_env(env,skip_render_check=True)


@pytest.mark.parametrize('suite',['standard','volatile','tight_budget'])
@pytest.mark.parametrize('policy',[spot_only,on_demand_only,checkpoint_aware])
def test_policies_obey_accounting_and_budget(suite,policy):
    for seed in range(4):
        result,_=episode(policy,seed,suite)
        assert result['spent']<=generate(seed,suite).budget
        assert result['completed']+result['missed']==12


def test_completion_at_deadline_is_valid():
    env=controlled(horizon=4,work=4)
    result=finish(env)
    assert result['completed']==1 and result['spent']==8


def test_guard_prevents_overspending():
    env=controlled(budget=1)
    result=finish(env)
    assert result['spent']==0 and result['completed']==0 and env.rejections>0


def test_checkpoint_survives_interruptions():
    env=controlled(horizon=10,work=5)
    risks=list(env.scenario.risks);risks[3]=1
    draws=list(env.scenario.draws);draws[3]=(0.,0.)
    env.scenario=replace(env.scenario,risks=tuple(risks),draws=tuple(draws))
    env.step(1);env.step(1);env.step(6)
    assert env.progress[0]==env.saved[0]==2
    env.step(1)
    assert env.progress[0]==2 and env.interruptions==1
    assert finish(env)['completed']==1


def test_interruption_during_checkpoint_does_not_save():
    env=controlled()
    risks=list(env.scenario.risks);risks[1]=1
    draws=list(env.scenario.draws);draws[1]=(0.,0.)
    env.scenario=replace(env.scenario,risks=tuple(risks),draws=tuple(draws))
    env.step(1);env.step(6)
    assert env.saved[0]==env.progress[0]==0 and env.lost==1
    finish(env)


def test_switching_instance_type_loses_unsaved_work():
    env=controlled();env.step(1);env.step(3)
    assert env.lost==1 and env.progress[0]==1
    finish(env,3)


def test_episode_boundary_and_bad_actions():
    env=controlled();finish(env)
    with pytest.raises(RuntimeError):env.step(1)
    env.reset(seed=8)
    with pytest.raises(ValueError):env.step(7)


def test_verifier_rejects_tampered_billing_work_and_truncated_evidence():
    env=controlled();finish(env)
    bad=copy.deepcopy(env.ledger);bad[0]['cost']=0
    assert not verify(env.scenario,bad)['success']
    bad=copy.deepcopy(env.ledger)
    for event in bad[0]['events']:
        if event['type']=='work':event['progress']=99
    assert not verify(env.scenario,bad)['success']
    assert not verify(env.scenario,env.ledger[:-1])['success']


def test_workload_and_trace_independent_of_policy():
    a,b=SpotEnv(),SpotEnv();a.reset(seed=3);b.reset(seed=3)
    for _ in range(10):a.step(2);b.step(4)
    assert a.scenario==b.scenario
    a.close();b.close()


def test_learner_updates_and_saves(tmp_path):
    agent,replay=DQN(7),Replay()
    rng=np.random.default_rng(5)
    for _ in range(100):
        obs=rng.random(24);replay.add(obs,1,1,obs,True)
    obs=np.ones(24);before=agent.values(obs)[1]
    for _ in range(100):assert np.isfinite(agent.learn(replay.sample(rng)))
    assert agent.values(obs)[1]>before
    agent.save(tmp_path/'policy.npz')
    np.testing.assert_array_equal(agent.values(obs),DQN.load(tmp_path/'policy.npz').values(obs))
    np.savez(tmp_path/'bad.npz',p0=np.zeros((24,64)),p1=np.zeros(64),p2=np.zeros((64,5)),p3=np.zeros(5))
    with pytest.raises(ValueError):DQN.load(tmp_path/'bad.npz')


def test_random_schedules_are_auditable_across_shifted_suites():
    for suite in ('standard','volatile','tight_budget'):
        for seed in range(20):
            rng=np.random.default_rng(seed+900)
            result,_=episode(lambda obs:int(rng.integers(7)),seed+800,suite)
            assert result['audit_success']


def test_checkpoint_without_workers_costs_nothing():
    env=controlled();env.step(6)
    assert env.spent==0 and env.progress[0]==0
    finish(env)


def test_checkpoint_loader_rejects_missing_arrays_and_nan(tmp_path):
    np.savez(tmp_path/'missing.npz',p0=np.zeros((24,64)))
    with pytest.raises(ValueError):DQN.load(tmp_path/'missing.npz')
    agent=DQN(7);agent.p[0][0,0]=np.nan;agent.save(tmp_path/'nan.npz')
    with pytest.raises(ValueError):DQN.load(tmp_path/'nan.npz')
