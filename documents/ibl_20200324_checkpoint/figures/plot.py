import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# plot experiment time scales
dt_per_trial = 10
trial_per_block = 50
dt_per_block = dt_per_trial * trial_per_block
block_per_session = 20
dt_per_session = dt_per_block * block_per_session
session_per_learning = 20000
dt_per_learning = dt_per_session * session_per_learning

dts_per_timescale = [dt_per_trial, dt_per_block, dt_per_session, dt_per_learning]
heights = [4, 3, 2, 1]
timescale_strs = ['Trial', 'Block', 'Session', 'Learning']

fig, ax = plt.subplots()
fig.suptitle('IBL Task Implementation Time Scales')
ax.set_xscale('log')
ax.set_xlabel('Number of Model Steps (dt)')
ax.set_yticklabels([])
for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_visible(False)
ax.barh(y=heights,
        height=0.5,
        width=dts_per_timescale,
        tick_label=timescale_strs)
plt.savefig('ibl_task_timescales.jpg')


# plot per-trial stimulus sampling
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
# correct side
ax = axes[1]
ax.set_title('Correct Side')
ax.set_xlim(-0.5, 2.)
for x in np.linspace(0, 1.5, 7):
    ax.axvline(x=x)
ax.axvline(x=1., color='r')
xs = np.linspace(-3., 3, 10000)
ax.plot(xs, norm.pdf(xs, loc=1., scale=1.), color='r')

# incorrect side
ax = axes[0]
ax.set_xlim(-0.5, 2.)
ax.set_title('Incorrect Side')
ax.axvline(x=0, color='r')
xs = np.linspace(-3., 3, 10000)
ax.plot(xs, norm.pdf(xs, loc=0., scale=1.), color='r')
plt.savefig('ibl_sampling_process.jpg')
