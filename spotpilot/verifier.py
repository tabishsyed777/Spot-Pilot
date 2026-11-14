"""Audit scheduling evidence without calling the environment transition function."""
from math import isclose


def verify(scenario, ledger):
    jobs = {j.id:j for j in scenario.jobs}
    progress = {j:0 for j in jobs}; saved = progress.copy()
    active, completed, expired = {}, set(), set()
    total = lost = interruptions = 0
    errors = []
    configs = {0:(),1:('spot',),2:('spot','spot'),3:('on',),4:('on','on'),5:('spot','on')}
    def require(condition,message):
        if not condition: raise ValueError(message)
    try:
        for t,entry in enumerate(ledger):
            require(entry['t']==t and t<scenario.horizon,'invalid time sequence')
            action=entry['action']; require(action in range(7),'invalid action')
            billed=0; pending={}; executed=set(); done_now=0; expired_now=0; lost_before=lost; rejected=False
            for e in entry['events']:
                kind=e['type']
                if kind=='budget_rejection':
                    requested=sum((scenario.prices[t] if w[1]=='spot' else scenario.on_demand_price)+(scenario.checkpoint_storage_price if action==6 else 0) for w in active.values())
                    require(not rejected and requested==e['requested_cost'] and requested>scenario.budget-total,'unjustified budget rejection')
                    rejected=True
                elif kind=='assign':
                    slot,jid=e['slot'],e['job']
                    require(slot in (0,1) and slot not in active and jid in jobs,'invalid assignment')
                    require(jid not in completed|expired and jid not in [w[0] for w in active.values()],'duplicate job allocation')
                    require(jobs[jid].arrival<=t<jobs[jid].deadline,'job not available')
                    require(action!=6 and slot<len(configs[action]) and configs[action][slot]==e['kind'],'assignment contradicts action')
                    active[slot]=(jid,e['kind'])
                elif kind=='execute':
                    slot,jid=e['slot'],e['job']
                    require(not rejected and slot not in executed and active.get(slot)==(jid,e['kind']),'invalid worker execution')
                    require(action==6 or (slot<len(configs[action]) and configs[action][slot]==e['kind']),'execution contradicts action')
                    require(e['mode']==('checkpoint' if action==6 else 'run'),'wrong execution mode')
                    require(jobs[jid].arrival<=t<jobs[jid].deadline,'execution outside job window')
                    failure=e['kind']=='spot' and scenario.draws[t][slot]<scenario.risks[t]
                    require(e['interrupted']==failure,'interruption evidence inconsistent with trace')
                    billed += (scenario.prices[t] if e['kind']=='spot' else scenario.on_demand_price)+(scenario.checkpoint_storage_price if action==6 else 0)
                    pending[slot]='interruption' if failure else ('save' if action==6 else 'work')
                    executed.add(slot)
                    interruptions+=int(failure)
                elif kind in ('work','save'):
                    slot,jid=e['slot'],e['job']
                    require(active.get(slot,(None,))[0]==jid and pending.pop(slot,None)==kind,'work/checkpoint without execution')
                    if kind=='work': progress[jid]+=1
                    else: saved[jid]=progress[jid]
                    require(progress[jid]<=jobs[jid].work and e['progress']==progress[jid],'invalid job progress')
                elif kind=='complete':
                    slot,jid=e['slot'],e['job']
                    require(active.get(slot,(None,))[0]==jid and jid not in completed and progress[jid]==jobs[jid].work,'invalid or repeated completion')
                    require(e['time']==t+1<=jobs[jid].deadline,'late completion')
                    completed.add(jid); done_now+=1; del active[slot]
                elif kind=='release':
                    slot,jid=e['slot'],e['job']
                    require(active.get(slot,(None,))[0]==jid,'release without allocation')
                    reason=e['reason']
                    if reason=='interruption': require(pending.pop(slot,None)=='interruption','unjustified interruption release')
                    elif reason=='budget': require(rejected,'release without budget rejection')
                    elif reason=='deadline': require(jobs[jid].deadline<=t+1,'premature expiration')
                    elif reason=='resize': require(action!=6 and (slot>=len(configs[action]) or configs[action][slot]!=active[slot][1]),'unnecessary resize release')
                    else: raise ValueError('unknown release reason')
                    loss=progress[jid]-saved[jid]
                    require(e['lost']==loss,'lost work mismatch')
                    lost+=loss; progress[jid]=saved[jid]; del active[slot]
                elif kind=='expire':
                    jid=e['job']
                    require(jid in jobs and jid not in completed|expired and jobs[jid].deadline<=t+1,'invalid expiration')
                    require(jid not in [w[0] for w in active.values()],'expired job still allocated')
                    expired.add(jid); expired_now+=1
                else: raise ValueError('unknown ledger event')
            require(not pending,'missing outcome for execution')
            require(all(slot in executed for slot in active),'unbilled active worker')
            require(all(j.deadline>t+1 or j.id in completed|expired for j in jobs.values()),'missing deadline outcome')
            require(entry['cost']==billed,'billing mismatch')
            total+=billed
            require(entry['spent']==total and total<=scenario.budget,'budget or running total violation')
            expected_reward=4*done_now-2*expired_now-.02*billed-.08*(lost-lost_before)
            require(isclose(entry['reward'],expected_reward,abs_tol=1e-8),'reward inconsistent with audited events')
        require(len(completed|expired)==len(jobs),'incomplete episode evidence')
        require(not active,'workers left allocated at episode end')
    except (ValueError,KeyError,TypeError,IndexError) as exc:
        errors.append(str(exc))
    return dict(success=not errors,reasons=errors,completed=len(completed),missed=len(expired),spent=total,lost_work=lost,interruptions=interruptions)
