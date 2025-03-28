import numpy as np
import configparser, ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pathlib
import subprocess
from source.commons import floor_decimal
from source.commons import writeFile
from matplotlib.patches import Rectangle
from scipy.stats import binomtest

def find_abs_state(state, stateLowerBound, stateResolution):
        # Find the abstract state of a continuous state
        return np.floor((state - stateLowerBound) // stateResolution).astype(int)

def if_within(state, lowerBound, upperBound, epsilon=1e-6):
        return np.all(state >= lowerBound + epsilon) and np.all(state <= upperBound - epsilon)

def convert_to_base3(i, stateDimension):
    base3 = []
    for _ in range(stateDimension):
        base3.append(i % 3)
        i //= 3
    return base3

class Abstraction:
    def __init__(self, dynamics, config_file):
        self.dynamics = dynamics

        config = configparser.ConfigParser()
        config.read(config_file)

        # Dimension of the state and control space
        self.stateDimension = int(config['DEFAULT']['stateDimension'])
        self.controlDimension = int(config['DEFAULT']['controlDimension'])
        
        # Domain of the state space
        self.stateLowerBound = self.parse_list(config['DEFAULT']['stateLowerBound'])
        self.stateUpperBound = self.parse_list(config['DEFAULT']['stateUpperBound'])

        # Domain of the control space
        self.controlLowerBound = self.parse_list(config['DEFAULT']['controlLowerBound'])
        self.controlUpperBound = self.parse_list(config['DEFAULT']['controlUpperBound'])

        # Domain of the goal set
        self.goalLowerBound = self.parse_list(config['DEFAULT']['goalLowerBound'])
        self.goalUpperBound = self.parse_list(config['DEFAULT']['goalUpperBound'])

        # Domain of the critical set
        self.criticalLowerBound = self.parse_list(config['DEFAULT']['criticalLowerBound'])
        self.criticalUpperBound = self.parse_list(config['DEFAULT']['criticalUpperBound'])

        # Resolution of the state space : size of each abstract cell
        self.stateResolution = self.parse_list(config['DEFAULT']['stateResolution'])

        # Number of samples in each abstract cell
        self.numObservations = int(config['DEFAULT']['numObservations'])
        self.numControlSamples = self.parse_list(config['DEFAULT']['numControlSamples']).astype(int)

        # Number of divisions in each dimension of the state space for when I want to find if there exists a control input such that the next state of the nominal system is inside the target set of an abstract state
        self.numVoxels = self.parse_list(config['DEFAULT']['numVoxels']).astype(int)

        # Number of abstract cells in each dimension
        self.absDimension = ((self.stateUpperBound - self.stateLowerBound + self.stateResolution - 1e-6) // self.stateResolution).astype(int)

        self.noiseLevel = float(config['DEFAULT']['noiseLevel'])
        self.lambdaValue = float(config['DEFAULT']['lambdaValue'])

        self.partition = {
            'state_variables': [f'x{i+1}' for i in range(self.stateDimension)],
            'dim': self.stateDimension,
            'lb': self.stateLowerBound,
            'ub': self.stateUpperBound,
            'regions_per_dimension': self.absDimension,
        }
        self.partition['size_per_region'] = self.stateResolution
        
        self.partition['goal_idx'] = set()
        self.partition['unsafe_idx'] = set()

        # iterate over all abstract states
        for abs_state_index, _ in np.ndenumerate(np.empty(self.absDimension, dtype=object)):
            abs_state_lower_bound = self.stateLowerBound + self.stateResolution * np.array(abs_state_index)
            abs_state_upper_bound = abs_state_lower_bound + self.stateResolution
            
            for i in range(len(self.goalLowerBound)):
                if self.if_within(abs_state_lower_bound, self.goalLowerBound[i], self.goalUpperBound[i]) and self.if_within(abs_state_upper_bound, self.goalLowerBound[i], self.goalUpperBound[i]):
                    self.partition['goal_idx'].add(abs_state_index)
            
            for i in range(len(self.criticalLowerBound)):
                if (np.all(abs_state_upper_bound > self.criticalLowerBound[i]) and
                    np.all(abs_state_lower_bound < self.criticalUpperBound[i])):
                    self.partition['unsafe_idx'].add(abs_state_index)
    
        print("goal_idx", self.partition['goal_idx'])
        print("unsafe_idx", self.partition['unsafe_idx'])
        # Every partition element also has an integer identifier
        iterator = itertools.product(*map(range, np.zeros(self.partition['dim'], dtype=int), self.partition['regions_per_dimension']))
        self.partition['tup2idx'] = {tup: idx for idx, tup in enumerate(iterator)}
        self.partition['idx2tup'] = {idx: tup for idx, tup in enumerate(iterator)}
        self.partition['nr_regions'] = len(self.partition['tup2idx'])

    def find_abs_state(self, state):
        # Find the abstract state of a continuous state
        return np.floor((state - self.stateLowerBound) / self.stateResolution).astype(int)

    def if_within(self, state, lowerBound, upperBound):
        return np.all(state >= lowerBound) and np.all(state <= upperBound)

    def parse_list(self, value):
        return np.array(ast.literal_eval(value))

    def generate_noisy_observations(self):
        # sample observations is an n-d array of lists, where each list contains the samples that reach the corresponding abstract state
        self.observations = np.empty(self.absDimension, dtype=object)

        for index, _ in np.ndenumerate(self.observations):
            self.observations[index] = []

        np.random.seed(42)

        x_obs = np.random.uniform(low=self.stateLowerBound, high=self.stateUpperBound, size=(self.numObservations, self.stateDimension))
        u_obs = np.random.uniform(low=self.controlLowerBound, high=self.controlUpperBound, size=(self.numObservations, self.controlDimension))
        
        control_res = (self.controlUpperBound - self.controlLowerBound) / self.numControlSamples
        control_indices = np.floor((u_obs - self.controlLowerBound) / control_res).astype(int)
        u_obs = self.controlLowerBound + control_indices * control_res + control_res / 2
        
        x_noisy = x_obs + np.random.normal(size=x_obs.shape) * self.noiseLevel
        u_noisy = u_obs + np.random.normal(size=u_obs.shape) * self.noiseLevel

        next_x = self.dynamics.vectorized_dynamics(x_noisy, u_noisy)
        abs_state_index = self.find_abs_state(x_obs)
        abs_state_tuples = [tuple(idx) for idx in abs_state_index]
        data = zip(abs_state_tuples, x_obs, u_obs, next_x)

        for abs_idx, state, control, next_state in tqdm(data, desc="processing samples", total=self.numObservations):
            self.observations[abs_idx].append((state, control, next_state))

        print("samples are processed")

        # index = (2,2)
        # plt.figure()
        # for i, record in enumerate(self.observations[index]):
        #     state = record[0]
        #     next_state = record[2]
        #     plt.scatter(state[0], state[1], c='b', s=2)
        #     plt.scatter(next_state[0], next_state[1], c='r', s=2)

        # plt.xlabel('State Dimension 1')
        # plt.ylabel('State Dimension 2')
        # plt.xticks(np.arange(self.stateLowerBound[0], self.stateUpperBound[0] + self.stateResolution[0], self.stateResolution[0]))
        # plt.yticks(np.arange(self.stateLowerBound[1], self.stateUpperBound[1] + self.stateResolution[1], self.stateResolution[1]))
        # plt.grid(True)
        # plt.savefig(f'plot{index}.png', dpi=500)

    def find_actions(self):
        self.actions = np.empty(self.absDimension, dtype=object)
        
        # Initialize all elements at once instead of inside the loop
        for index, _ in np.ndenumerate(self.actions):
            self.actions[index] = []
        
        # Process with vectorized operations where possible
        for index, _ in tqdm(np.ndenumerate(self.actions), desc="Finding actions", total=np.prod(self.absDimension)):
            obs = np.array(self.observations[index])
            next_abs_states = self.find_abs_state(obs[:, 2, :])
            unique_states, counts = np.unique(next_abs_states, axis=0, return_counts=True)
            valid_indices = counts >= len(obs) * self.lambdaValue

            freq_actions = unique_states[valid_indices].tolist()
            for i in range(len(freq_actions)):
                if np.all(np.array(freq_actions[i]) >= 0) and np.all(np.array(freq_actions[i]) < self.absDimension) and not np.all(np.array(index) == np.array(freq_actions[i])):
                    self.actions[index].append(freq_actions[i])
        print("actions are found")

    def find_transition(self, abs_index, inverse_confidence):
        # Pre-allocate arrays for interface data
        interface = np.empty(np.array([len(self.actions[abs_index]), *self.numVoxels]), dtype=object)
        interface_fitness = np.ones(np.array([len(self.actions[abs_index]), *self.numVoxels]), dtype=int) * 1e3
        for idx in np.ndindex(interface.shape):
            interface[idx] = []
        
        # Calculate bounds and resolution once
        abs_state_lower_bound = self.stateLowerBound + self.stateResolution * np.array(abs_index)
        control_res = (self.controlUpperBound - self.controlLowerBound) / self.numControlSamples

        voxcon_obs = np.empty(np.array([*self.numVoxels, *self.numControlSamples]), dtype=object)
        for idx in np.ndindex(voxcon_obs.shape):
            voxcon_obs[idx] = []
        
        # Vectorize operations on observations
        obs_array = np.array(self.observations[abs_index])

        voxel_indices = np.floor((obs_array[:, 0, :] - abs_state_lower_bound) / (self.stateResolution/self.numVoxels)).astype(int)
        control_indices = np.floor((obs_array[:, 1, :] - self.controlLowerBound) / control_res).astype(int)

        for i in range(len(obs_array)):
            voxel_idx = tuple(voxel_indices[i])
            control_idx = tuple(control_indices[i])
            idx = voxel_idx + control_idx
            voxcon_obs[idx].append((obs_array[i, 0], obs_array[i, 1], obs_array[i, 2]))

        for index, obs_list in np.ndenumerate(voxcon_obs):
            if len(obs_list) == 0:
                continue
            obs_array = np.array(obs_list)[:, 2, :]
            
            # abs_next_indexes = np.floor((obs_array[:, 2, :] - self.stateLowerBound) / self.stateResolution).astype(int)

            voxel_index = index[:self.stateDimension]
            control_index = index[self.stateDimension:]

            for i, action_target in enumerate(self.actions[abs_index]):
                target_center = self.stateLowerBound + np.array(action_target) * self.stateResolution + self.stateResolution / 2
                # matches = np.sum(np.all(abs_next_indexes == action_target_array, axis=1))
                fitness = np.mean(np.sum((obs_array - target_center)**2, axis=1))
                if fitness < interface_fitness[tuple([i, *voxel_index])]:
                    interface_fitness[tuple([i, *voxel_index])] = fitness
                    interface[tuple([i, *voxel_index])] = control_index

        print(self.actions[abs_index])
        for action_index in range(len(self.actions[abs_index])):
            voxel_indices = np.ndindex(tuple(self.numVoxels))

            roi = {-1, -2}
            action = tuple(self.actions[abs_index][action_index])

            # if action != (6, 6):
            #     continue

            for i in range(3**self.stateDimension):
                dim = np.array(convert_to_base3(i, self.stateDimension))
                offset = np.array([0] * self.stateDimension)
                offset[dim == 2] = 1
                offset[dim == 1] = -1

                neighbor = np.array(action) + np.array(offset)
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in self.partition['tup2idx']:
                    roi.add(self.partition['tup2idx'][neighbor_tuple])

            results = []
            for voxel_index in voxel_indices:
                control_index = tuple(interface[action_index, *voxel_index])
                voxcon_index = tuple([*voxel_index, *control_index])
                
                # Fast calculation of voxel bounds
                voxel_lb = self.stateLowerBound + np.array(abs_index) * self.stateResolution + np.array(voxel_index) * self.stateResolution / self.numVoxels
                voxel_ub = voxel_lb + self.stateResolution / self.numVoxels
                
                control_bar = self.controlLowerBound + np.array(control_index) * control_res + control_res / 2

                count_lb = np.zeros(self.partition['nr_regions'] + 2, dtype=int)
                count_ub = np.zeros(self.partition['nr_regions'] + 2, dtype=int)
                Qlb = []
                Qub = []

                for (state, control, next_state) in voxcon_obs[voxcon_index]:
                    dx = np.maximum(np.abs(voxel_lb - state), np.abs(voxel_ub - state))
                    du = np.abs(control_bar - control)
                    radius = dx @ self.dynamics.L_X(state, control).T + du @ self.dynamics.L_U(state, control).T

                    lb = next_state - radius - 1e-8
                    ub = next_state + radius + 1e-8
                    Qlb.append(lb)
                    Qub.append(ub)

                    abs_lb = self.find_abs_state(lb)
                    abs_ub = self.find_abs_state(ub)

                    tuples = set(itertools.product(*map(range, abs_lb, abs_ub + 1)))
                    
                    intersections = set()
                    for index in tuples:
                        if np.all(np.array(index) >= 0) and np.all(np.array(index) < self.absDimension):
                            if index in self.partition['goal_idx']:
                                intersections.add(-1)
                            elif index in self.partition['unsafe_idx'] or self.partition['tup2idx'][index] not in roi:
                                intersections.add(-2)
                            else:
                                intersections.add(self.partition['tup2idx'][index])
                        else:
                            intersections.add(-2)

                    if len(intersections) == 1:
                        count_lb[list(intersections)[0]] += 1
                        count_ub[list(intersections)[0]] += 1
                    elif len(intersections) > 1:
                        for ind in intersections:
                            count_ub[ind] += 1

                prob_intervals = {}

                for i in roi:
                    binom = binomtest(k=count_lb[i], n=len(voxcon_obs[voxcon_index]))
                    probs = binom.proportion_ci(confidence_level = 1 - inverse_confidence)
                    low =   probs.low
                    
                    binom = binomtest(k=count_ub[i], n=len(voxcon_obs[voxcon_index]))
                    probs = binom.proportion_ci(confidence_level = 1 - inverse_confidence)
                    high =  probs.high

                    prob_intervals[i] = [low, high]
                
                # Qlb = np.array(Qlb)
                # Qub = np.array(Qub)
                # plt.figure()
                # for i, record in enumerate(voxcon_obs[voxcon_index]):
                #     state = record[0]
                #     next_state = record[2]
                #     plt.scatter(state[0], state[1], c='b', s=1)
                #     plt.scatter(next_state[0], next_state[1], c='r', s=1)
                #     rectangle = plt.Rectangle(Qlb[i, :], Qub[i, 0] - Qlb[i, 0], Qub[i, 1] - Qlb[i, 1], fill=True, edgecolor='none', facecolor='g', alpha=0.1)
                #     plt.gca().add_patch(rectangle)
                # plt.xlabel('State Dimension 1')
                # plt.ylabel('State Dimension 2')
                # plt.xticks(np.arange(self.stateLowerBound[0], self.stateUpperBound[0] + self.stateResolution[0], self.stateResolution[0]))
                # plt.yticks(np.arange(self.stateLowerBound[1], self.stateUpperBound[1] + self.stateResolution[1], self.stateResolution[1]))
                # plt.grid(True)
                # plt.savefig(f'plot{voxcon_index}.png', dpi=500)

                results.append(prob_intervals)
                # print(voxcon_index, prob_intervals)

            prob_intervals = {}
            for i in roi:
                prob_intervals[i] = (1, 0)
                for result in results:
                    prob_intervals[i] = [min(prob_intervals[i][0], result[i][0]), max(prob_intervals[i][1], result[i][1])]
            # print("----------\n", prob_intervals)
            self.transitions[abs_index].append(prob_intervals)
            # return

    def find_transitions(self, inverse_confidence = 0.05/50000):
        self.transitions = np.empty(self.absDimension, dtype=object)
        for index, _ in tqdm(np.ndenumerate(self.transitions), desc="Finding transitions", total=np.prod(self.absDimension)):
            self.transitions[index] = []

            if index in self.partition['goal_idx']:
                # print(' ---- Skip',index,'because it is a goal region')
                continue
            if index in self.partition['unsafe_idx']:
                # print(' ---- Skip',index,'because it is a critical region')
                continue
            
            self.find_transition(index, inverse_confidence)

    def create_IMDP(self, foldername, timebound=np.inf, problem_type='reachavoid'):
        print('\nExport abstraction as PRISM model...')

        timespec = ""
        if timebound != np.inf:
            timespec = 'F<=' + str(timebound) + ' '
        else:
            timespec = 'F '

        if problem_type == 'avoid':
            specification = 'Pminmax=? [' + timespec + ' "failed" ]'
        else:
            specification = 'Pmaxmin=? [' + timespec + ' "reached" ]'

        self.specification = specification
        # Write specification file
        writeFile(foldername + "/abstraction.pctl", 'w', specification)

        ##############################

        # Define tuple of state variables (for header in PRISM state file)
        state_var_string = ['(' + ','.join([f'x{i+1}' for i in range(self.stateDimension)]) + ')']

        state_file_header = ['0:(' + ','.join([str(-2)] * self.stateDimension) + ')',
                             '1:(' + ','.join([str(-1)] * self.stateDimension) + ')']

        state_file_content = []

        for abs_state in itertools.product(*map(range, self.absDimension)):
            state_id = self.partition['tup2idx'][abs_state] + 2
            state_representation = str(state_id) + ':' + str(abs_state).replace(' ', '')
            state_file_content.append(state_representation)

        state_file_string = '\n'.join(state_var_string + state_file_header + state_file_content)

        # Write content to file
        writeFile(foldername + "/abstraction.sta", 'w', state_file_string)

        label_head = ['0="init" 1="deadlock" 2="reached" 3="failed"'] + \
                     ['0: 1 3'] + ['1: 2']

        label_body = ['' for i in range(self.transitions.size)]

        for abs_state in itertools.product(*map(range, self.absDimension)):
            state_id = self.partition['tup2idx'][abs_state]
            substring = str(state_id + 2) + ': 0'

            # Check if region is a deadlock state
            if self.actions[abs_state] == []:
                substring += ' 1'

            # Check if region is in goal set
            if abs_state in self.partition['goal_idx']:
                substring += ' 2'
            elif abs_state in self.partition['unsafe_idx']:
                substring += ' 3'

            label_body[state_id] = substring

        label_full = '\n'.join(label_head) + '\n' + '\n'.join(label_body)

        # Write content to file
        writeFile(foldername + "/abstraction.lab", 'w', label_full)

        ##############################

        nr_choices_absolute = 0
        nr_transitions_absolute = 0
        transition_file_list = ''
        head = 2

        for index, _ in np.ndenumerate(self.transitions):
            if index in self.partition['goal_idx']:
                # print(' ---- Skip',index,'because it is a goal region')
                continue
            if index in self.partition['unsafe_idx']:
                # print(' ---- Skip',index,'because it is a critical region')
                continue
            
            if self.transitions[index] != []:
                choice = -1
                for action_ind in range(len(self.transitions[index])):
                    choice += 1
                    nr_choices_absolute += 1
                    for next_state_idx, next_state_prob in self.transitions[index][action_ind].items():
                        prob_str =  '[' + str(max(next_state_prob[0], 1e-8)) + ',' + str(next_state_prob[1]) + ']'

                        transition_file_list += str(self.partition['tup2idx'][index] + head) + ' ' + str(choice) + ' ' + \
                        str(next_state_idx + head) + ' ' + str(prob_str) + \
                        ' a_' + str(int(self.partition['tup2idx'][tuple(self.actions[index][action_ind])]) + head) + '\n'
                        
                        nr_transitions_absolute += 1
                
            else: 
                new_transitions = str(self.partition['tup2idx'][index] + head) + ' 0 ' + str(self.partition['tup2idx'][index] + head) + ' [1.0,1.0]\n'
                transition_file_list += new_transitions
                nr_choices_absolute += 1
                nr_transitions_absolute += 1
                    
        
        size_states = self.transitions.size + head
        size_choices = nr_choices_absolute + head
        size_transitions = nr_transitions_absolute + head
        header = str(size_states) + ' ' + str(size_choices) + ' ' + str(size_transitions) + '\n'
        firstrow = '0 0 0 [1.0,1.0]\n1 0 1 [1.0,1.0]\n'
        writeFile(foldername + "/abstraction.tra", 'w', header + firstrow + transition_file_list)

    def solve_iMDP(self, foldername, prism_executable):
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
        
        print('Starting PRISM...')
        
        with open(foldername + "/abstraction.pctl", 'r') as file:
            spec = file.read().strip()
        mode = "interval"
        
        print(' -- Running PRISM with specification for mode',
              mode.upper()+'...')

        file_prefix = foldername + "PRISM_" + mode
        policy_file = file_prefix + '_policy.txt'
        vector_file = file_prefix + '_vector.csv'

        options = ' -exportstrat "' + policy_file + '"' + \
                  ' -exportvector "' + vector_file + '"'
    
        print(' --- Execute PRISM command for EXPLICIT model description')        


        prism_java_memory = 8
        prism_executable = prism_executable
        prism_file = foldername + '/abstraction.all'

        model_file      = '"'+prism_file+'"'
        # Check if the prism executable can be found and if so, run it on the generated iMDP.
        if not pathlib.Path(prism_executable).is_file():
            raise Exception(f"Could not find the prism executable. Please check if the following path to the executable is correct: {str(prism_executable)}")
        command = prism_executable + " -javamaxmem " + \
                  str(prism_java_memory) + "g -importmodel " + model_file + " -pf '" + \
                  spec + "' " + options
        
        subprocess.Popen(command, shell=True).wait()
