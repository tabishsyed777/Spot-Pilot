"""Regenerate publication charts from the recorded benchmark; requires matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
data=json.loads((root/'artifacts/benchmark.json').read_text())
labels={'spot_only':'Spot only','on_demand_only':'On-demand only','deadline_aware':'Deadline heuristic','checkpoint_aware':'Checkpoint heuristic','double_dqn_1':'Double DQN · seed 7','double_dqn_2':'Double DQN · seed 17','double_dqn_3':'Double DQN · seed 27'}
fig,axes=plt.subplots(3,2,figsize=(13,12),layout='constrained')
for row,(suite,policies) in enumerate(data['suites'].items()):
    for col,(metric,title) in enumerate([('completion_rate','Jobs completed by deadline (%)'),('spent','Mean synthetic credits spent')]):
        ax=axes[row,col];values=[policies[k]['summary'][metric]*(100 if metric=='completion_rate' else 1) for k in labels]
        colors=['#416a92' if not k.startswith('double_dqn') else '#7857b8' for k in labels]
        bars=ax.barh(list(labels.values()),values,color=colors,height=.65);ax.invert_yaxis()
        ax.set_title(suite.replace('_',' ').title()+' — '+title,fontsize=11)
        if metric=='completion_rate':ax.set_xlim(0,100)
        else:
            budget=150 if suite=='tight_budget' else 220
            ax.set_xlim(0,budget+30);ax.axvline(budget,color='#c04f44',linestyle='--',linewidth=1,label='Budget');ax.legend(fontsize=8,loc='lower right')
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
fig.suptitle('SpotPilot: three trained policies against conventional schedulers\n100 paired scenarios per policy per suite · lower spend is useful only alongside completed work',fontsize=14)
fig.savefig(root/'artifacts/benchmark.png',dpi=160)
plt.close(fig)
