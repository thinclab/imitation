import copy
import logging

import gym
import pandas as pd
from gym import spaces
# from gym.utils import seeding
from time import sleep, time
from enum import *

import pickle

import numpy as np
import random
import pickle
import math
import os

from imitation.envs.airl_envs.half_cheetah_mdfd_weights import SharedRobustAIRL_Stuff

# path = os.path.dirname (os.path.realpath (__file__))
# rollout_path = os.path.abspath(os.path.join(path, os.pardir, os.pardir, os.pardir, os.pardir))

logger = logging.getLogger(__name__)

class HumanSortingContinuous3D(gym.Env,SharedRobustAIRL_Stuff):
    """
    This environment is a slightly modified version of sorting env in this paper: MMAP-BIRL(https://arxiv.org/pdf/2109.07788.pdf).
    As opposed to the paper, this environment has continuous state and action spaces and works in 3D space. 
    In order to learn the sorting behavior, the learner observes an expert (a human) performing the sort individually 
    and tries to find an approxiamte reward function that best describes the expert's intents. Using the estimated reward function, 
    the learner can find the optimal policy using RL.
    ------------------------------------------------------------------------------------------------------------------------
    Current State: S
    Next State: S'
    Action: A
    Transitions: T = Pr(S' | S, A)
    Reward: R(S,A) 
    ------------------------------------------------------------------------------------------------------------------------
    State and action spaces are as below: 
    J: Joint, O: Onion, G: Glove, C: Conveyor belt, B: Bin, H: Head
    x: normalized x coordinate of the object in the image
    y: normalized y coordinate of the object in the image
    conf: prediction confidence of the object
    l: Object's label
    d: delta (difference)
    *********************
    s_agent - (J6x, J6y, J6z, J8x, J8y, J8z, J10x, J10y, J10z, 
               O1x, O1y, O1z, O1l, ..., O5x, O5y, O5z, O5l)
    a_agent - (dJ1x, dJ1y, dJ1conf, ..., dJ17x, dJ17y, dJ17conf)
    ------------------------------------------------------------------------------------------------------------------------
    J6: coordinates of the right shoulder of the expert's body.
    J8: coordinates of the right elbow of the expert's body.
    J10: coordinates of the right wrist of the expert's body.
    Oix, Oiy, Oiz, Oil: x, y, and z coordinates of the center of the ith onion and it's label (blemished (0.0) or unblemished (1.0)).
    
    Gx, Gy, Gz, Gl: x, y, and z coordinates of the center of the glove and it's label.
    Cx, Cy, Cz, Cl: x, y, and z coordinates of the center of the Conveyor belt and it's label.
    Bx, By, Bz, Bl: x, y, and z coordinates of the center of the Bin and it's label.
    Hx, Hy, Hz, Hl: x, y, and z coordinates of the center of the Head and it's label.
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    dJix, dJiy, dJiz: x, y, and z difference between next state and current state (next_state(x, y, z) - current_state(x, y, z))
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    Episode starts from one of the valid start states where onion is on conv.
    Episode ends when all onions are successfully picked, inspected, and placed.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    # Ok
    def __init__(self, rollout_path, full_observable=True, max_steps=100,
        cov_diag_val_transition_model = 0.0001, 
        cov_diag_val_st_noise = 0.1,
        cov_diag_val_act_noise = 0.1, 
        noise_insertion=False):
        
        self.startStateCSV       = [f'{rollout_path}/csvs/start_state_{i}.csv' for i in [1]] # range(1, 11)]
        
        print(f'self.startStateCSV :: {self.startStateCSV}')
        print(f'self.startStateCSV length:: {len(self.startStateCSV)}')
        
        
        self.statesCSV           = [f'{rollout_path}/csvs/states_{i}.csv' for i in [1]] # range(1, 11)]
        # self.actionsCSV          = f'{rollout_path}/csvs/actions.csv'
        # self.conveyorBoxCSV      = f'{rollout_path}/csvs/conveyor_box_xy.csv' #TODO
        self.minMaxStateCSV      = f'{rollout_path}/csvs/min_max_states.csv'
        # self.minMaxActionCSV     = f'{rollout_path}/csvs/min_max_actions.csv'
        # self.minMaxStartStateCSV = f'{rollout_path}/csvs/min_max_start_state.csv'
        self.kptModelFiles       = [f'{rollout_path}/regression_models/kpt_model_{i}.sav' for i in [6, 8]]

        self.kptModels           = [pickle.load(open(kpt_model, 'rb')) for kpt_model in self.kptModelFiles]

        self.startState          = [pd.read_csv(self.startStateCSV[i],       header=None).values.tolist() for i in [0]] # range(0, 10)]
        
        self.startState          = list(np.array(self.startState).reshape(-1, 29))
        
        print(f'self.startState :: {self.startState}')
        
        print('self.startState.shape ::', np.array(self.startState).shape)
        
        # self.states              = pd.read_csv(self.statesCSV,           header=None).values.tolist()
        # self.actions             = pd.read_csv(self.actionsCSV,          header=None).values.tolist()

        # self.minStartState       = pd.read_csv(self.minMaxStartStateCSV, header=None).iloc[0, 1:].values.reshape(-1,).tolist()
        # self.maxStartState       = pd.read_csv(self.minMaxStartStateCSV, header=None).iloc[1, 1:].values.reshape(-1,).tolist()
        
        self.minState            = pd.read_csv(self.minMaxStateCSV,      header=None).iloc[0, 1:].values.reshape(-1,).tolist()
        self.maxState            = pd.read_csv(self.minMaxStateCSV,      header=None).iloc[1, 1:].values.reshape(-1,).tolist()
        # self.minAction           = pd.read_csv(self.minMaxActionCSV,     header=None).iloc[0, 1:].values.reshape(-1,).tolist()
        # self.maxAction           = pd.read_csv(self.minMaxActionCSV,     header=None).iloc[1, 1:].values.reshape(-1,).tolist()

        #TODO
        # self.conveyor            = pd.read_csv(self.conveyorBoxCSV,      header=None).iloc[0, 1:].values.reshape(-1,).tolist()
        # self.box                 = pd.read_csv(self.conveyorBoxCSV,      header=None).iloc[1, 1:].values.reshape(-1,).tolist()

        self.conveyor = [0, 0]
        self.box = [0, 0]

        self._max_episode_steps  = max_steps
        self.step_cost           = 0.0
        self.state_params_count  = 29 # 3 joints (each with 3 coords) + 5 onions (each with 3 coords and 1 label)
        self.action_params_count = 3 # dx, dy, and dz of the right hand wrist
        self.min_dist_to_grasp   = 0.05 #TODO
        self.reward              = self.step_cost # Why reward and step cost are equal?
        self.full_observable     = full_observable
        self._step_count         = None 
        self._full_obs           = None
        self.prev_obsv           = None
        self._agent_dones        = None #??
        self.steps_beyond_done   = None 

        self.INVALID_STATE       = np.array([float('-inf')] * self.state_params_count)

        self.p_i                 = float('inf')
        self.n_i                 = float('-inf')

        self.maxState = [x for x in self.maxState]
        self.minState = [x for x in self.minState]

        self._obs_high         = np.array(self.maxState) 
        self._obs_low          = np.array(self.minState) 

        for i in range(29): #TODO
            # if i == 6 or i == 7, or i==8: # skip the wrist
            #     continue
            
            self._obs_high[i]  = self.p_i
            self._obs_low[i]   = self.n_i

        self._obs_high         = np.round(self._obs_high, 2)
        self._obs_low          = np.round(self._obs_low,  2)

        # self._act_high         = np.array([0.09] * self.action_params_count)  
        # self._act_low          = np.array([-0.09] * self.action_params_count)  
        
        self._act_high         = np.array([1.0]  * self.action_params_count) #TODO
        self._act_low          = np.array([-1.0] * self.action_params_count) #TODO

        self.observation_space = spaces.Box(self._obs_low, self._obs_high)
        self.action_space      = spaces.Box(self._act_low, self._act_high)

        self.seed()

        self.num_samples = 100000 # TODO
        self.obs_size = self.observation_space.shape[0]
        self.act_size = self.action_space.shape[0]
        self.noise_insertion = noise_insertion
        # need to be repeated for this class
        self.cov_diag_val_st_noise = cov_diag_val_st_noise
        self.cov_diag_val_act_noise = cov_diag_val_act_noise
        
        SharedRobustAIRL_Stuff.__init__(self,\
            cov_diag_val_transition_model = cov_diag_val_transition_model, 
            cov_diag_val_st_noise = cov_diag_val_st_noise,
            cov_diag_val_act_noise = cov_diag_val_act_noise,
            obs_size = self.obs_size,
            act_size = self.act_size,
            intended_next_state_f = self.intended_next_state)

    # Done
    def get_reward(self, env_state, action):
        '''
        @brief Provides reward for the desired and undesired behaviors.
        '''

        next_env_state       = self.findNxtState(state=env_state, action=action)
        _, current_onions    = self.disassembleStateParams(env_state)
        _, next_onions       = self.disassembleStateParams(next_env_state)
        
        current_onion, index = self.get_onion_in_focus(env_state, current_onions)
        next_onion           = next_onions[index-1]
        
        current_onion_x , current_onion_y, current_onion_z, current_onion_label = current_onion[0], current_onion[1], current_onion[2], current_onion[3]
        next_onion_x,     next_onion_y,    next_onion_z, next_onion_label    = next_onion[0],    next_onion[1],    next_onion[2], next_onion[3]
        
        
        if current_onion_z < 0 or current_onion_label < 0:
            self.reward -= 0.001
        
        # if onion is good
        elif current_onion_label > 0:
            
            # if action is to throw at the bin:  -1 
            if next_onion_label < 0 or next_onion_x < 0 or next_onion_y < 0 or next_onion_z < 0:
                self.reward -= 1
                self.reward -= 0.001
                
            # if action is to place on conveyor: +1
            elif current_onion_y < -0.2 and next_onion_x >= 0.6 and next_onion_x <= 0.9 and next_onion_z < 0.8 and next_onion_y < -0.4:
                self.reward += 1
                self.reward -= 0.001
            
            else:
                self.reward -= 0.001
        
        # if onion is bad: 
        elif current_onion_label == 0:
            
            # if action is to throw at the bin:  +1 
            if next_onion_label < 0 or next_onion_x < 0 or next_onion_y < 0 or next_onion_z < 0:
                self.reward += 1
                self.reward -= 0.001
                
            # if action is to place on conveyor: -1
            elif current_onion_y < -0.2 and next_onion_x >= 0.6 and next_onion_x <= 0.9 and next_onion_z < 0.8 and next_onion_y < -0.4:
                self.reward -= 1
                self.reward -= 0.001
            
            else:
                self.reward -= 0.001
        
        else:
            self.reward -= 0.001
                    
    # Done
    def get_3d_dist(self, point1, point2):
        
        # Calculate the distance
        distance = math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2) + ((point2[2] - point1[2]) ** 2))
        
        return distance

    # K-Done
    def get_closest_onion_to_gripper(self, onions, gripper):
        dist = 100
        closest_onion = -1
        
        onions = np.array(onions).reshape(-1, 4)
        gripper = list(np.array(gripper).ravel())
        
        # gripper[0] = gripper[0] * 2.0 #TODO
        # gripper[1] = gripper[1] * 2.0 #TODO
        # gripper[2] = gripper[2] * 2.0 #TODO
        
        for i in range(onions.shape[0]):
            onion = [onions[i, 0], onions[i, 1], onions[i, 2]]
            
            dist_to_gripper = self.get_3d_dist(onion, gripper)
            
            # print(f'dist of onion {i+1} to gripper: {dist_to_gripper}')
            
            if dist_to_gripper < dist:
                dist = dist_to_gripper
                closest_onion_num = i+1
        
        return dist, closest_onion_num
    
    # K-Done
    def get_onion_in_focus(self, env_state, onions):
        
        env_state = list(np.array(env_state).ravel())
        onions    = list(np.array(onions).ravel())
        
        # if something is grasped, the onion in focus is the grasped onion
        gripper_dist_to_onion, closest_onion = self.get_closest_onion_to_gripper(onions, gripper = [env_state[6], env_state[7], env_state[8]])
        
        if gripper_dist_to_onion < self.min_dist_to_grasp and label >= -0.5:
            onions = list(np.array(onions).reshape(-1, 4))
            return onions[closest_onion - 1] , closest_onion
        
        for i , onion in enumerate(list(np.array(onions).reshape(-1, 4))):
            
            x, y, z, label = onion[0], onion[1], onion[2], onion[3]
            
            if x < 0 or z < 0: #TODO
                continue
            
            if y < -22 or label < 0:
                continue
            
            return onion, i+1

        return onions[0], 1
    
    # Done
    def disassembleStateParams(self, state):
        
        state = list(np.array(state).ravel())
        
        kpts = np.asarray(state[:9]).reshape(-1, 3)
        onions = np.asarray(state[9:]).reshape(-1, 4)
                
        return kpts, onions
    
    # Should be fine (didn't change it but should work)
    def runRegressionModel(self, reg_model, x):
        # load the model from disk
        x = np.array(x).reshape(1, -1)
        y_pred = reg_model.predict(x)
        
        return y_pred 
    
    # Done
    def set_prev_obsv(self, state):
        self.prev_obsv = copy.copy(state)

    # Done
    def get_prev_obsv(self):
        return self.prev_obsv

    # Done
    def get_invalid_state(self):
        return self.INVALID_STATE

    # Done
    def get_init_obs(self, fixed_init=True):
        '''
        @brief - Samples from the start state distrib and returns a joint one hot obsv.
        # '''
        
        if fixed_init:
            init_obs = self.startState[0][0]
        
        else:
            init_obs = self.sample_start()
        
        self.set_prev_obsv(init_obs)

        return np.array(init_obs)
       
    # Should be OK
    def tune_actions(self, action_list, env_state):
        
        env_state = list(np.array(env_state).ravel())
        action_list = list(np.array(action_list).ravel())
        
        for i in range (len(action_list)):
            new_state_i = action_list[i] + env_state[i]
            
            if new_state_i > self._obs_high[i]:
                action_list[i] -= (new_state_i - self._obs_high[i])
                
            if new_state_i < self._obs_low[i]:
                action_list[i] += (self._obs_low[i] - new_state_i)

        return action_list
    
    # Done
    def get_keypoint_features(self, target_keypoint_index):
        if target_keypoint_index == 8:
            return [10]
        elif target_keypoint_index == 6:
            return [8, 10]
        elif target_keypoint_index == 5:
            return [6, 10]
        elif target_keypoint_index == 0:
            return [5, 6, 10]
        elif target_keypoint_index == 1:
            return [0, 5, 6, 10]
        elif target_keypoint_index == 2:
            return [0, 5, 6, 10]
        elif target_keypoint_index == 3:
            return [0, 5, 6, 10]
        elif target_keypoint_index == 4:
            return [0, 5, 6, 10]
        elif target_keypoint_index == 7:
            return [5]
        elif target_keypoint_index == 9:
            return [7]
        elif target_keypoint_index == 12:
            return [5, 6, 10]
        elif target_keypoint_index == 11:
            return [5, 12]
        elif target_keypoint_index == 13:
            return [11]
        elif target_keypoint_index == 14:
            return [12]
        elif target_keypoint_index == 15:
            return [13]
        elif target_keypoint_index == 16:
            return [14]
        
        return []

    # Done
    def get_sampled_onion_label(self):
        
        sample_label = round(random.uniform(0, 1))
                   
        return sample_label
    
    # Done
    def isValidState(self, state, num_params=None):
        '''
        @brief - Checks if a given state is valid or not.

        '''
        
        state = np.array(state).ravel()
        
        if num_params is None:
            num_params = self.state_params_count
        
        if state is None:
            return False
        
        if state.shape[0] != num_params:
            print('State length is not correct ...')
            return False
        
        for i in range(num_params):
            
            st = float(state[i])
                
            if st < self._obs_low[i]:
                
                if i < 34:
                    print(f'state index {i} has an invalid value: {st}')
                    print(f'*** Value is less than the minimum limit which is {self._obs_low[i]}')
                    return False

                elif i % 3 != 0 and st != -1.0:
                    print(f'state index {i} has an invalid value: {st}')
                    print(f'*** Value is less than the minimum limit which is {self._obs_low[i]}')
                    return False
                
                elif i % 3 == 0 and st != -0.2:
                    print(f'state index {i} has an invalid value: {st}')
                    print(f'*** Value is less than the minimum limit which is {self._obs_low[i]}')
            
            elif st > self._obs_high[i]:
                print(f'state index {i} has an invalid value: {st}')
                print(f'*** Value is greater than maximum limit which is {self._obs_high[i]}')
                return False
        
        return True
    
    # Done
    def isValidStartState(self, state):
        '''
        @brief - Checks if a given state is a valid start state or not.

        '''
        
        state = list(np.array(state).ravel())
        
        # check if the state is valid in general
        if (not self.isValidState(state)):
            return False
        
        # check if the state has conditions of a legal starting state
        # make sure human is not touching any onion
        onions = np.array(state[9:]).reshape(-1, 4)

        gripper_dist_to_onion, closest_onion = self.get_closest_onion_to_gripper(onions, gripper = [state[6], state[7], state[8]])
        
        if gripper_dist_to_onion < self.min_dist_to_grasp:
            return False
        
        # if the state is not problematic (i.e. invalid) --> return true
        return True

    # Done
    def isValidAction(self, state, action, num_params=None):
        '''
        @brief - For each state there are a few invalid actions, returns only valid actions.
        '''
        
        action = list(np.array(action).ravel())
        
        if num_params is None:
            num_params = self.action_params_count
        
        if len(action) != num_params:
            print(f'action: {action}')
            # print(f'action[0][0]: ', action[0][0])
            print(f'Action length is {len(action)} which is not compatible with the number of params which is {num_params} ...')
            return False
        
        for i in range(num_params):
            
            if action[i] < self._act_low[i]:
                print(f'action parameter {i} is {action[i]} which is less than the minimum range of {self._act_low[i]}')
                return False
            
            elif action[i] > self._act_high[i]:
                print(f'action parameter {i} is {action[i]} which is higher than the maximum range of {self._act_high[i]}')
                return False

        next_state = self.findNxtState(state, action)
        
        return self.isValidState(next_state)

    #TODO
    def convert_env_state_to_visual_state(self, env_state):
        
        env_state = list(np.array(env_state).ravel())
        
        vis_state = [0] * 94
        
        for i in range(94):
            if i == 0 :
                vis_state[i] = 'unknown'
            
            elif i >= 1 and i <= 51:
                if i % 3 == 1:
                    kpt_index = int(i / 3)
                    vis_state[i] = env_state[2 * kpt_index]              
                elif i % 3 == 2:
                    kpt_index = int(i / 3)
                    vis_state[i] = env_state[(2 * kpt_index) + 1]
                else:
                    vis_state[i] = 1.0
            
            elif i >= 52 and i <= 81:
                vis_state[i] = env_state[i - 18]

            # glove/gripper
            elif i >= 82 and i <= 84:
                if i == 82:
                    vis_state[i] = vis_state[30]
                elif i == 83:
                    vis_state[i] = vis_state[31]
                else:
                    vis_state[i] = 0.4
            
            # conveyor
            elif i >= 85 and i <= 87:
                if i == 85:
                    vis_state[i] = self.conveyor[0]
                elif i == 86:
                    vis_state[i] = self.conveyor[1]
                else:
                    vis_state[i] = 0.6
            
            # bin
            elif i >= 88 and i <= 90:
                if i == 88:
                    vis_state[i] = self.box[0]
                elif i == 89:
                    vis_state[i] = self.box[1]
                else:
                    vis_state[i] = 0.8
            
            # head
            elif i >= 91 and i <= 93:
                if i == 91:
                    vis_state[i] = (vis_state[9] + vis_state[12]) / 2.0
                elif i == 92:
                    vis_state[i] = (vis_state[10] + vis_state[13]) / 2.0
                else:
                    vis_state[i] = 1.0
            
        return vis_state

    #TODO
    def convert_all_vis_states(self, trajs):
        data = trajs
        obs = data.obs
        acts = data.acts

        viz_states = []
        for ob in obs:
            viz_states.append(self.convert_env_state_to_visual_state(ob.tolist()))
        
        return viz_states 

    # Done
    def sample_start(self, num_params=None):
    
        random.seed(time())
        
        if num_params is None:
            num_params = self.state_params_count
        
        k = random.randint(0, 9)
        sample_state = self.startState[k][0]
            
        for i in range(num_params):
            if i >= 9 and i % 4 == 0:
                sample_state[i] = self.get_sampled_onion_label()
        
        # sample_state = np.array(sample_state)
        
        return sample_state # numpy array with shape: (1, 29)
    
    # Done (for 3 joints)
    def get_joint_movements(self, env_state, action):
    
        env_state = list(np.array(env_state).ravel())
        action = list(np.array(action).ravel())

        action_list = [0] * 9
        
        action_list[6] = action[0]
        action_list[7] = action[1]
        action_list[8] = action[2]
        
        kpts_list = [0, 1]
        
        for i in kpts_list: 
                        
            x = []
            
            x.append(env_state[6] + action[0])
            x.append(env_state[7] + action[1])
            x.append(env_state[8] + action[2])
                       
            coord = self.runRegressionModel(self.kptModels[i], x)
            
            action_list[i * 2] = list(coord.ravel())[0] - env_state[i * 2]
            action_list[(i * 2) + 1] = list(coord.ravel())[1] - env_state[(i * 2) + 1]
            action_list[(i * 2) + 2] = list(coord.ravel())[2] - env_state[(i * 2) + 2]

        action_list = self.tune_actions(action_list, env_state)

        # Print the action list
        # print('action_list: ', action_list)
        
        return action_list
    
    # def get_joint_movements(self, env_state, action):
    
    #     env_state = list(np.array(env_state).ravel())
    #     action = list(np.array(action).ravel())

    #     action_list = [0] * 34
        
    #     action_list[20] = action[0]
    #     action_list[21] = action[1]
        
    #     kpts_list = [8, 6, 5, 0, 1, 2, 3, 4, 7, 9, 12, 11, 13, 14, 15, 16]
        
    #     for i in kpts_list: 
            
    #         feature_indices = self.get_keypoint_features(target_keypoint_index=i)
    #         x = []
    #         for _, fi in enumerate(feature_indices):
    #             new_x = env_state[fi * 2] + action_list[fi * 2]
    #             new_y = env_state[(fi * 2) + 1] + action_list[(fi * 2) + 1]
    #             x.append(new_x)
    #             x.append(new_y)
            
    #         coord = self.runRegressionModel(self.kptModels[i], x)
            
    #         action_list[i * 2] = list(coord.ravel())[0] - env_state[i * 2]
    #         action_list[(i * 2) + 1] = list(coord.ravel())[1] - env_state[(i * 2) + 1]

    #     action_list = self.tune_actions(action_list, env_state)

    #     # Print the action list
    #     # print('action_list: ', action_list)
        
    #     return action_list
    
    # Done
    def findNxtState(self, state, action):
        ''' 
        @brief - Returns the valid nextstates. 
        NOTE: 
        This function assumes that you're doing the action from the appropriate current state.
        Inappropriate actions are filtered out by getValidActions method now. 

        Keypoints: {x, y, z} * 3
        Onions: {x, y, z, label} * 5
        
        action: x, y, z movement value for the human wrist
        '''
        # len = 9 , len= 20 
        kpts_state, Onions = self.disassembleStateParams(state)
        # len = 9
        kpts_action = self.get_joint_movements(env_state=state, action=action)
        
        # len = 20
        Onions = np.array(Onions).reshape(-1, 4)
        # len = 9 (3*3)    
        new_kpts = list(np.add(np.array(kpts_state).ravel(), np.array(kpts_action).ravel())) 
        
        next_gripper = list(np.array(new_kpts).ravel())[6:9]
        gripper      = list(np.array(kpts_state).ravel())[6:9]
        
        next_dist,    next_closest_onion_index    = self.get_closest_onion_to_gripper(Onions, next_gripper)
        current_dist, current_closest_onion_index = self.get_closest_onion_to_gripper(Onions, gripper)
        
        closest_onion_x, closest_onion_y, closest_onion_z = Onions[current_closest_onion_index - 1, 0], Onions[current_closest_onion_index - 1, 1], Onions[current_closest_onion_index - 1, 2]
        
        # print('current closest onion: ', current_closest_onion_index)
        # print('next closest onion: '   , next_closest_onion_index)
        
        # by default, don't change the location and label of onions and don't change the the gripper's status
        new_onions = list(np.array(Onions).ravel())
        
        # grasped_onion_x, grasped_onion_y, grasped_onion_z = new_kpts[6] * 2, new_kpts[7] * 2, new_kpts[8] * 2
        grasped_onion_x, grasped_onion_y, grasped_onion_z = new_kpts[6], new_kpts[7], new_kpts[8]
        
        # if trying to put on conveyor but not put the onion on it yet
        # if closest_onion_y < -0.2 and closest_onion_z > 0.85:
        #     pass
        
        # if something is grasped now, check if it is in the bin area or good onion location or somewhere else
        if current_dist < self.min_dist_to_grasp:
            
            # if action is to throw at bin, vanish the onion
            if (closest_onion_x > 0.3 and closest_onion_x < 0.6 and closest_onion_y > 0.45 and closest_onion_z > 0.95):
                new_onions[(current_closest_onion_index  - 1) * 4     ] = -1.0
                new_onions[((current_closest_onion_index - 1) * 4) + 1] = -1.0
                new_onions[((current_closest_onion_index - 1) * 4) + 2] = -1.0
                new_onions[((current_closest_onion_index - 1) * 4) + 3] = -1.0
                
            # if action is to place on conveyor, change z of the onion to be on the conveyor
            elif closest_onion_x > 0.6 and closest_onion_x < 0.85 and \
                 closest_onion_y < -0.4 and closest_onion_z < 0.85:
                new_onions[(current_closest_onion_index  - 1) * 4]      = closest_onion_x
                new_onions[((current_closest_onion_index - 1) * 4) + 1] = closest_onion_y
                new_onions[((current_closest_onion_index - 1) * 4) + 2] = 0.77
                
            else: 
                new_onions[(current_closest_onion_index  - 1) * 4]      = grasped_onion_x
                new_onions[((current_closest_onion_index - 1) * 4) + 1] = grasped_onion_y
                new_onions[((current_closest_onion_index - 1) * 4) + 2] = grasped_onion_z
        
        elif next_dist < self.min_dist_to_grasp and grasped_onion_y >= -0.2 and grasped_onion_y <= 0.2 : # and grasped_onion_y <= 0.7:
            
            new_onions[(next_closest_onion_index  - 1) * 4]      = grasped_onion_x
            new_onions[((next_closest_onion_index - 1) * 4) + 1] = grasped_onion_y
            new_onions[((next_closest_onion_index - 1) * 4) + 2] = grasped_onion_z
        
        new_state = np.concatenate((np.array(new_kpts).ravel(), np.array(new_onions).ravel()), axis=0)
        
        new_state = np.round(new_state, 2)
        
        # return list(new_state.ravel())
        return new_state.ravel()
    
    # Done    
    def intended_next_state(self, observation, action):
        return self.findNxtState(state=observation, action=action)

    # Done
    def reset(self, fixed_init=True):
        '''
        @brief - Just setting all params to defaults and returning a valid start obsv.
        '''
        self._step_count = 0
        self.reward = self.step_cost

        self._agent_dones = False
        self.steps_beyond_done = None
        
        return self.get_init_obs(fixed_init)

    # Done
    def step(self, action, verbose=1):
        '''
        @brief - Performs given actions and returns next observation, reward, and done
        '''
        # action[0] *= 0.09
        # action[1] *= 0.09
                
        self._step_count += 1
        self.reward = self.step_cost
        
        # if verbose:
        #     state = self.prev_obsv

        state = self.prev_obsv
        
        nxt_s = None
        
        if np.array_equal(state, self.INVALID_STATE):
            return self.get_invalid_state(), self.reward, self._agent_dones, {}
        
        if self.isValidState(state=self.prev_obsv):
            if self.isValidAction(state=self.prev_obsv, action=action):
                nxt_s = self.findNxtState(state=self.prev_obsv, action=action)
            else:
                if verbose:
                    logger.error(f"Step {self._step_count}: Invalid action: {action}, in current state: {state} can't transition anywhere else with this. Staying put and ending episode!")
                    
                    # self.write_invalid_sa_in_file(state, action)
                    
                self._agent_dones = True

                ''' Sending all invalid actions to an impossible sink state '''

                return self.get_invalid_state(), self.reward, self._agent_dones, {}
            
        else:
            if verbose:
                logger.error(f"Step {self._step_count}: Invalid current state {state}, ending episode!")
            self._agent_dones = True
            raise ValueError

        self.get_reward(self.prev_obsv, action)

        self.set_prev_obsv(state=nxt_s)

        if self._step_count >= self._max_episode_steps:
            self._agent_dones = True

        if self.all_onions_are_sorted():
            self._agent_dones = True
            
        # if self.reward < self.step_cost:
        #     self._agent_dones = True

        if self.steps_beyond_done is None and self._agent_dones:
            self.steps_beyond_done = 0
            
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warning(
                    f"Step {self._step_count}: You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(done) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.reward = 0
                    
        return nxt_s, self.reward, self._agent_dones, {}

    # Done
    def all_onions_are_sorted(self):
        
        _, Onions = self.disassembleStateParams(self.prev_obsv)
        
        Onions = np.array(Onions).reshape(-1, 4)
        
        for i in range(Onions.shape[0]):
            
            onion_x     = Onions[i, 0]
            onion_y     = Onions[i, 1]
            onion_z     = Onions[i, 2]
            onion_label = Onions[i, 3]
            
            if onion_y >= -0.2 or (onion_y < -0.2 and onion_z > 0.8):
                return False
            
        return True
    