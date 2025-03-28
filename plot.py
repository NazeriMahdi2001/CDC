import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import configparser, ast
from models.car2d import Robot2D

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size' : 18,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

policy = {}
with open('outputPRISM_interval_policy.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split('=')
        value = value.split('_')[1]
        key = tuple(ast.literal_eval(key))
        policy[key] = int(value)

trans = {}
with open('output/abstraction.sta', 'r') as file:
    next(file)  # Skip the header line "(x1,x2)"
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = int(key)
            value = tuple(ast.literal_eval(value))
            trans[key] = value

def parse_list(value):
    return np.array(ast.literal_eval(value))

config_file = 'models/car2d.conf'
system = Robot2D()

config = configparser.ConfigParser()
config.read(config_file)

# Dimension of the state and control space
stateDimension = int(config['DEFAULT']['stateDimension'])
controlDimension = int(config['DEFAULT']['controlDimension'])

# Domain of the state space
stateLowerBound = parse_list(config['DEFAULT']['stateLowerBound'])
stateUpperBound = parse_list(config['DEFAULT']['stateUpperBound'])

# Domain of the goal set
goalLowerBound = parse_list(config['DEFAULT']['goalLowerBound'])
goalUpperBound = parse_list(config['DEFAULT']['goalUpperBound'])

# Domain of the critical set
criticalLowerBound = parse_list(config['DEFAULT']['criticalLowerBound'])
criticalUpperBound = parse_list(config['DEFAULT']['criticalUpperBound'])

stateResolution = parse_list(config['DEFAULT']['stateResolution'])

prismIntervalVector = np.genfromtxt('outputPRISM_interval_vector.csv')
prismIntervalVector = prismIntervalVector[2:]
absDimension = ((stateUpperBound - stateLowerBound + stateResolution - 1e-6) // stateResolution).astype(int)
prismIntervalVector = prismIntervalVector.reshape(absDimension).T


plt.imshow(prismIntervalVector, extent=(stateLowerBound[0], stateUpperBound[0], stateUpperBound[1], stateLowerBound[1]), aspect='auto', cmap='coolwarm', vmin=0.0, vmax=1.0)
plt.colorbar()

# plt.xlabel(r'$x$', fontsize=16)
# plt.ylabel(r'$v$', fontsize=16, rotation='horizontal')

plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16, rotation='horizontal')

# plt.xlabel(r'$\theta$', fontsize=16)
# plt.ylabel(r'$\omega$', fontsize=16, rotation='horizontal')

plt.minorticks_on()
plt.tick_params(which='both', width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=5)

plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=6))
plt.gca().yaxis.set_major_locator(plt.LinearLocator(numticks=6))

plt.gca().xaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[0], stateUpperBound[0], stateResolution[0])))
plt.gca().yaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[1], stateUpperBound[1], stateResolution[1])))

plt.grid(which='both', linestyle='--', linewidth=0.05)

plt.gca().invert_yaxis()
plt.savefig('prism_interval_vector_heatmap.pdf', dpi=500, bbox_inches='tight')

# plt.figure(figsize=[10, 10])
# plt.xticks(np.arange(stateLowerBound[0], stateUpperBound[0] + stateResolution[0], stateResolution[0]))
# plt.yticks(np.arange(stateLowerBound[1], stateUpperBound[1] + stateResolution[1], stateResolution[1]))
# plt.xlim(stateLowerBound[0], stateUpperBound[0])
# plt.ylim(stateLowerBound[1], stateUpperBound[1])
# plt.xlabel(r'$X_1$')
# plt.ylabel(r'$X_2$')
# plt.grid(True)

x = np.array([0.43, 1.34])

init = tuple(((x - stateLowerBound) // stateResolution).astype(int))
for _ in range(100):
    if init in policy:
        next = policy[init]
        if next in trans:
            next = trans[next]
            # draw an arrow from init to next
            a = stateLowerBound + np.array(init) * stateResolution + stateResolution / 2
            b = stateLowerBound + np.array(next) * stateResolution + stateResolution / 2
            plt.arrow(a[0], a[1], b[0] - a[0], b[1] - a[1], head_width=0.2, head_length=0.05, fc='b', ec='b').set_zorder(10)
            init = next

# add the goal set
goalLowerBound = np.array(goalLowerBound)
goalUpperBound = np.array(goalUpperBound)
plt.fill_between([goalLowerBound[0][0], goalUpperBound[0][0]], goalLowerBound[0][1], goalUpperBound[0][1], color='g', alpha=0.2)

for i in range(len(criticalLowerBound)):
    clb = np.array(criticalLowerBound[i])
    cub = np.array(criticalUpperBound[i])
    plt.fill_between([clb[0], cub[0]], clb[1], cub[1], color='r', alpha=0.2)

plt.savefig('trajectory.pdf', dpi=500, bbox_inches='tight')

plt.figure(figsize=[10, 10])
plt.xlim(stateLowerBound[0], stateUpperBound[0])
plt.ylim(stateLowerBound[1], stateUpperBound[1])

plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16, rotation='horizontal')

# plt.xlabel(r'$x$', fontsize=16)
# plt.ylabel(r'$v$', fontsize=16, rotation='horizontal')

# plt.xlabel(r'$\theta$', fontsize=16)
# plt.ylabel(r'$\omega$', fontsize=16, rotation='horizontal')

init = tuple(((x - stateLowerBound) // stateResolution).astype(int))
print(prismIntervalVector[init])

# plt.scatter(x[0], x[1], c='black', s=50, marker='*').set_zorder(11)
# voxelResolution = stateResolution / numDivisions
# for i in range(100):
#     if init in policy:
#         next = policy[init]
#         if next in trans:
#             next = trans[next]
#             # draw an arrow from init to next

#             policy_filename = f'policy/policy_{init}_{next}.npy'
#             refined_policy = np.load(policy_filename)

#             residue = x - init * stateResolution - stateLowerBound
#             ind = (residue // voxelResolution).astype(int)
#             control = refined_policy[*ind, :]
            
#             nx = system.set_state(*x).update_dynamics(control) + np.random.uniform(-0.5*stateResolution*noiseLevel, 0.5*stateResolution*noiseLevel, stateDimension)
#             # nx = system.set_state(*x).update_dynamics(control) + np.random.normal(scale=noiseLevel, size=stateDimension)

#             if i > 0:
#                 plt.scatter(x[0], x[1], c='navy', s=25, marker="X").set_zorder(11)
#             plt.arrow(x[0], x[1], nx[0] - x[0], nx[1] - x[1], width=0.01, head_width=0, head_length=0, fc='navy', ec='navy').set_zorder(10)
            
#             init = tuple(((nx - stateLowerBound) // stateResolution).astype(int))
#             x = nx

# plt.scatter(x[0], x[1], c='black', s=50, marker='s').set_zorder(11)
# # add the goal set
# goalLowerBound = np.array(goalLowerBound)
# goalUpperBound = np.array(goalUpperBound)
# plt.fill_between([goalLowerBound[0], goalUpperBound[0]], goalLowerBound[1], goalUpperBound[1], color='g', alpha=0.3, linewidth=0.0)

# for i in range(len(criticalLowerBound)):
#     clb = np.array(criticalLowerBound[i])
#     cub = np.array(criticalUpperBound[i])
#     plt.fill_between([clb[0], cub[0]], clb[1], cub[1], color='r', alpha=0.3, linewidth=0.0)

# plt.minorticks_on()
# plt.tick_params(which='both', width=1)
# plt.tick_params(which='major', length=7)
# plt.tick_params(which='minor', length=4)

# plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=5))
# plt.gca().yaxis.set_major_locator(plt.LinearLocator(numticks=6))

# plt.gca().xaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[0], stateUpperBound[0], stateResolution[0])))
# plt.gca().yaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[1], stateUpperBound[1], stateResolution[1])))

# plt.grid(which='both', linestyle='--', linewidth=0.1)

# plt.gca().set_xticklabels([])
# plt.gca().set_yticklabels([])

# plt.savefig('trajectory_refined.pdf', dpi=500, bbox_inches='tight')