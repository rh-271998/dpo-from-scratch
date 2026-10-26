import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../result/eval_metrics_granular.csv')

# ---- aggregate across domains, per model ----
agg = df.groupby('model')[['held', 'cave', 'concede', 'stubborn', 'good', 'bad', 'other']].sum()

def rate(num, den):
    return 100 * num / den

hold = {m: rate(agg.loc[m, 'held'], agg.loc[m, 'held'] + agg.loc[m, 'cave']) for m in agg.index}
concede = {m: rate(agg.loc[m, 'concede'], agg.loc[m, 'concede'] + agg.loc[m, 'stubborn']) for m in agg.index}

BASE, DPO = '#9aa5b1', '#2a9d8f'     # muted gray-blue vs confident teal

# ===== Figure 1: calibration =====
fig, ax = plt.subplots(figsize=(6.5, 4.2))
groups = ['Resist wrong pushback\n(hold-rate)', 'Accept right pushback\n(concede-rate)']
base_vals = [hold['Base'], concede['Base']]
dpo_vals = [hold['DPO'], concede['DPO']]
x = np.arange(len(groups))
w = 0.36
b1 = ax.bar(x - w / 2, base_vals, w, label='Base', color=BASE)
b2 = ax.bar(x + w / 2, dpo_vals, w, label='DPO (ours)', color=DPO)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f'{r.get_height():.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 112)
ax.set_ylabel('% of committed responses')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10)
ax.set_title('DPO yields calibrated deference', fontsize=13, fontweight='bold', pad=12)
ax.legend(frameon=False, loc='lower right')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color='#e6e6e6')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('../result/fig_calibration.png', dpi=200)

# ===== Figure 2: outcome composition by domain =====
GOOD, BAD, OTHER = '#2a9d8f', '#e76f51', '#c9ccd1'
order = [('arithmetic', 'Base'), ('arithmetic', 'DPO'), ('fact', 'Base'), ('fact', 'DPO')]
labels = ['Arith\nBase', 'Arith\nDPO', 'Fact\nBase', 'Fact\nDPO']
tot = {k: df[(df.domain == k[0]) & (df.model == k[1])][['good', 'bad', 'other']].sum() for k in order}
g = [100 * tot[k]['good'] / tot[k].sum() for k in order]
b = [100 * tot[k]['bad'] / tot[k].sum() for k in order]
o = [100 * tot[k]['other'] / tot[k].sum() for k in order]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
x = np.arange(len(order))
ax.bar(x, g, 0.6, label='Correct behavior', color=GOOD)
ax.bar(x, b, 0.6, bottom=g, label='Sycophantic / stubborn', color=BAD)
ax.bar(x, o, 0.6, bottom=np.array(g) + np.array(b), label='Other / unparsed', color=OTHER)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('% of responses')
ax.set_ylim(0, 100)
ax.set_title('DPO collapses sycophancy; new degeneration only on arithmetic',
             fontsize=11.5, fontweight='bold', pad=12)
ax.legend(frameon=False, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=3)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('../result/fig_composition.png', dpi=200, bbox_inches='tight')

print('hold-rate   :', {m: round(hold[m], 1) for m in hold})
print('concede-rate:', {m: round(concede[m], 1) for m in concede})
print('saved fig_calibration.png and fig_composition.png')
