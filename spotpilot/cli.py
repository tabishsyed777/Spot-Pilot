import argparse
import json
from pathlib import Path
import platform
import numpy as np
import gymnasium
from .agent import DQN, Replay
from .env import SpotEnv, FEATURES, ACTION_NAMES
from .evaluation import episode, evaluate
from . import baselines
from .worker import demo as worker_demo


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')


def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    agent,replay=DQN(args.seed),Replay();history=[];best=-float('inf');selected=0;episode_id=0
    with SpotEnv(record=False) as env:
        obs,_=env.reset(seed=args.seed)
        for step in range(1,args.steps+1):
            epsilon=max(.08,1-step/(args.steps*.75));action=agent.act(obs,epsilon)
            ns,r,term,trunc,_=env.step(action);replay.add(obs,action,r,ns,term or trunc);obs=ns
            if replay.size>=512 and step%4==0:agent.learn(replay.sample(agent.rng))
            if term or trunc:
                episode_id+=1;obs,_=env.reset(seed=args.seed+episode_id)
            if step%10000==0 or step==args.steps:
                validation=evaluate(agent.act,20,50000,audit=False)['summary'];validation.update(step=step,epsilon=epsilon);history.append(validation)
                if validation['return_']>best:best=validation['return_'];selected=step;agent.save(out/'policy.npz')
                print(json.dumps(validation),flush=True)
    agent.save(out/'last_policy.npz')
    write_json(out/'training.json',dict(seed=args.seed,steps=args.steps,selected_step=selected,selection='maximum mean return over 20 standard validation seeds 50000..50019',training_suite='standard',features=FEATURES,actions=ACTION_NAMES,learning_rate=.0007,discount=.98,replay_capacity=30000,batch_size=64,target_period=250,learning_starts=512,update_every=4,epsilon_floor=.08,epsilon_decay_fraction=.75,python=platform.python_version(),numpy=np.__version__,gymnasium=gymnasium.__version__,history=history))


def benchmark(args):
    policies={name:getattr(baselines,name) for name in ('idle','spot_only','on_demand_only','deadline_aware','checkpoint_aware')}
    for i,path in enumerate(args.models):policies[f'double_dqn_{i+1}']=DQN.load(path).act
    results=dict(protocol=dict(episodes=args.episodes,seed_base=args.seed_base,models=args.models,suite_seed_offsets=dict(standard=0,volatile=100000,tight_budget=200000)),suites={})
    for suite,offset in results['protocol']['suite_seed_offsets'].items():
        results['suites'][suite]={name:evaluate(policy,args.episodes,args.seed_base+offset,suite) for name,policy in policies.items()}
        print(json.dumps(dict(suite=suite,summary={k:v['summary'] for k,v in results['suites'][suite].items()})),flush=True)
    write_json(args.output,results)


def main():
    p=argparse.ArgumentParser(description='SpotPilot: RL for cloud scheduling under a hard budget');sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--steps',type=int,default=80000);t.add_argument('--seed',type=int,default=7);t.add_argument('--output',default='runs/train');t.set_defaults(fn=train)
    b=sub.add_parser('benchmark');b.add_argument('--models',nargs='+',default=['artifacts/seed7/policy.npz']);b.add_argument('--episodes',type=int,default=100);b.add_argument('--seed-base',type=int,default=300000);b.add_argument('--output',default='runs/benchmark.json');b.set_defaults(fn=benchmark)
    d=sub.add_parser('demo');d.add_argument('--model',default='artifacts/seed7/policy.npz');d.add_argument('--seed',type=int,default=700001);d.add_argument('--suite',choices=['standard','volatile','tight_budget'],default='standard');d.add_argument('--output',default='runs/demo.json')
    def run_demo(args):
        result,evidence=episode(DQN.load(args.model).act,args.seed,args.suite);write_json(args.output,dict(result=result,**evidence));print(json.dumps(result,indent=2))
    d.set_defaults(fn=run_demo)
    w=sub.add_parser('worker-demo');w.add_argument('--output',default='runs/worker');w.set_defaults(fn=lambda args:print(json.dumps(worker_demo(args.output),indent=2)))
    args=p.parse_args()
    if getattr(args,'steps',1)<1 or getattr(args,'episodes',1)<1:p.error('steps and episodes must be positive')
    try:args.fn(args)
    except (FileNotFoundError,FileExistsError,ValueError) as exc:p.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
