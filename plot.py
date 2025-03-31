import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import configparser, ast
from models.car2d import Robot2D
import pickle

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

tup2idx = {}
idx2tup = {}
with open('output/abstraction.sta', 'r') as file:
    next(file)  # Skip the header line "(x1,x2)"
    for line in file:
        line = line.strip()
        if line:
            key, value = line.split(':')
            key = int(key)
            value = tuple(ast.literal_eval(value))
            tup2idx[value] = key
            idx2tup[key] = value

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
numVoxels = parse_list(config['DEFAULT']['numVoxels']).astype(int)
noiseLevel = float(config['DEFAULT']['noiseLevel'])

prismIntervalVector = np.genfromtxt('outputPRISM_interval_vector.csv')
prismIntervalVector = prismIntervalVector[2:]
absDimension = ((stateUpperBound - stateLowerBound + stateResolution - 1e-6) // stateResolution).astype(int)
prismIntervalVector = prismIntervalVector.reshape(absDimension).T


plt.imshow(prismIntervalVector, extent=(stateLowerBound[0], stateUpperBound[0], stateUpperBound[1], stateLowerBound[1]), aspect='auto', cmap='coolwarm', vmin=0.0, vmax=1.0)
plt.colorbar()

plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$y$', fontsize=20, rotation='horizontal')

plt.minorticks_on()
plt.tick_params(which='both', width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=5)

plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=4))
plt.gca().yaxis.set_major_locator(plt.LinearLocator(numticks=4))

plt.gca().xaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[0], stateUpperBound[0], stateResolution[0])))
plt.gca().yaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[1], stateUpperBound[1], stateResolution[1])))

plt.grid(which='both', linestyle='--', linewidth=0.05)

plt.gca().invert_yaxis()
plt.savefig('prism_interval_vector_heatmap.pdf', dpi=500, bbox_inches='tight')


plt.figure(figsize=[10, 10])
plt.xlim(stateLowerBound[0], stateUpperBound[0])
plt.ylim(stateLowerBound[1], stateUpperBound[1])

plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$y$', fontsize=20, rotation='horizontal')

start = np.array([0.43, 0.34])
init = tuple(((start - stateLowerBound) // stateResolution).astype(int))
print(prismIntervalVector[init])

plt.scatter(start[0], start[1], c='black', s=50, marker='s').set_zorder(11)

for _ in range(100):
    current = init
    x = start
    for i in range(100):
        if current in policy:
            next = policy[current]
            next = idx2tup[next]
            if next in tup2idx:
                # draw an arrow from current to next
                policy_filename = f'controller/{tup2idx[current] - 2}.bin'
                with open(policy_filename, 'rb') as f:
                    refined_policy = pickle.load(f)

                residue = x - current * stateResolution - stateLowerBound
                voxel_ind = (residue // (stateResolution / numVoxels)).astype(int)

                u = refined_policy[tuple([*voxel_ind, *next])]
                
                noisy_x = x + np.random.normal(scale=noiseLevel, size=stateDimension)
                noisy_u = u + np.random.normal(scale=noiseLevel, size=controlDimension)
                nx = system.set_state(*noisy_x).update_dynamics(noisy_u)

                plt.arrow(x[0], x[1], nx[0] - x[0], nx[1] - x[1], width=0.001, head_width=0, head_length=0, fc='grey', ec='grey', alpha=0.2).set_zorder(10)
                
                current = tuple(((nx - stateLowerBound) // stateResolution).astype(int))
                x = nx
    plt.scatter(x[0], x[1], c='black', s=5, marker=".", alpha=0.5).set_zorder(11)

# plt.scatter(x[0], x[1], c='black', s=50, marker='s').set_zorder(11)

# add the goal set
goalLowerBound = np.array(goalLowerBound)
goalUpperBound = np.array(goalUpperBound)
plt.fill_between([goalLowerBound[0][0], goalUpperBound[0][0]], goalLowerBound[0][1], goalUpperBound[0][1], color='g', alpha=0.2)

for i in range(len(criticalLowerBound)):
    clb = np.array(criticalLowerBound[i])
    cub = np.array(criticalUpperBound[i])
    plt.fill_between([clb[0], cub[0]], clb[1], cub[1], color='r', alpha=0.2)

plt.minorticks_on()
plt.tick_params(which='both', width=1)
plt.tick_params(which='major', length=7)
plt.tick_params(which='minor', length=4)

plt.gca().xaxis.set_major_locator(plt.LinearLocator(numticks=4))
plt.gca().yaxis.set_major_locator(plt.LinearLocator(numticks=4))

plt.gca().xaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[0], stateUpperBound[0], stateResolution[0])))
plt.gca().yaxis.set_minor_locator(plt.FixedLocator(np.arange(stateLowerBound[1], stateUpperBound[1], stateResolution[1])))

plt.grid(which='both', linestyle='--', linewidth=0.1)

plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])

plt.savefig('trajectory_refined.pdf', dpi=500, bbox_inches='tight')
