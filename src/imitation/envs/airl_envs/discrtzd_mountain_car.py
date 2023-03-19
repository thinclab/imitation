from gym.envs.classic_control.mountain_car import MountainCarEnv
from numpy import *
from gym import Env, spaces
import numpy as np
import math 

class DiscretizedStateMountainCarEnv(MountainCarEnv):
    '''
    Version of classic mountain car 
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py 
    with following changes:
        > state space with each double bounded dimension divided in D intervals, 
            creating D^2 multidimensional partitions of state space. these partitions are enumerated by ascending indices 
        > method that finds the index of the partition an input state belong to 
        > method that samples a state from a partition corresponding to input index 
        > transition model method that uses Gaussian distribution to compute state transition probabilities 
        > observation or noise insertion model that moves cart position by 10% of positionSpan/D 

    '''
    def __init__(self, D=10, n_smp_pr_dim=20):
        '''
        D: number of partitions of each double bounded dimension of state
        n_smp_pr_dim: number of discrete samples per dimension to be used in estimating value of a continuous integral 
        '''
        super(DiscretizedStateMountainCarEnv, self).__init__()
        self._D = D
        self._n_smp_pr_dim = n_smp_pr_dim 

        self.num_dims = 2 # position and velocity 

        # create enumerated Box spaces to be used as partitions 
        self._obs_sp_partitions = []
        
        low_array = self.observation_space.low
        high_array = self.observation_space.high
        
        # order in state: position, speed

        low_position = self.min_position
        low_array[0] = low_position
        for i in range(1,D+1):
            high_position = low_position + (self.max_position -self.min_position)/D
            high_array[0] = high_position

            low_speed = -self.max_speed
            low_array[1] = low_speed
            for i in range(1,D+1):
                high_speed = low_speed + (self.max_speed-(-self.max_speed))/D
                high_array[1] = high_speed
                # create box
                self._obs_sp_partitions.append(spaces.Box(low_array, high_array, dtype=np.float32))
                
                # for next iteration
                low_speed = high_speed
                low_array[1] = low_speed
            
            # for next iteration
            low_position = high_position 
            low_array[0] = low_position

        self.insertNoiseprob = 0.99
        self.percChangeInPos = 200

        size_parts = [] 
        for part_ind in range(len(self._obs_sp_partitions)):
            _, size = self.discrete_samples_to_estimate_integral(part_ind)
            size_parts.append(size) 
        
        state_space_size = self.state_space_size()
        # sum_size_parts_rd = np.round(sum(size_parts),3) 
        # writing hack below as round function didn't work as expected. this hack applies to only this state space and these partitions. 
        sum_size_parts_rd = float(str(sum(size_parts))[:len(str(0.001))])

        assert (state_space_size == sum_size_parts_rd),f"size of whole space {state_space_size} should be sum of parts {sum_size_parts_rd}"

    def find_partition(self,s):
        # finds the index of the partition an input state belong to 
        for ind, bx_sp in enumerate(self._obs_sp_partitions):
            # check membership of state in a partition
            if bx_sp.contains(s):
                return ind
    
    def sample_random_state_from_partition(self, ind):
        # uniformly distributed sample from a partition corresponding to input index 
        low_array = self._obs_sp_partitions[ind].low
        high_array = self._obs_sp_partitions[ind].high

        return np.random.uniform(low_array,high_array,(self.num_dims,))

    def discrete_samples_to_estimate_integral(self, ind):
        # return evenly spaced or uniformly distributed self._n_smp_pr_dim number of states 
        # for estimating an integral over input partition
        # alos, returns the size of partition 

        evenly_spaced = False
        list_states = []

        low_array = self._obs_sp_partitions[ind].low
        high_array = self._obs_sp_partitions[ind].high

        if evenly_spaced:
            # equal distance per dimension
            position = self._obs_sp_partitions[ind].low[0]
            for i in range(1,self._n_smp_pr_dim):
                speed = self._obs_sp_partitions[ind].low[1]

                for i in range(1,self._n_smp_pr_dim):
                    list_states.append(np.array([position, speed]))
                    speed += i*(high_array[1]-low_array[1])/self._n_smp_pr_dim
                
                position += i*(high_array[0]-low_array[0])/self._n_smp_pr_dim
        else:
            # uniformly sampled 
            for i in range(self._n_smp_pr_dim**self.num_dims):
                list_states.append(self.sample_random_state_from_partition(ind))

        return list_states,np.prod(high_array-low_array)

    def step_sa(self, s, a): # AdversarialTrainer > train_disc uses this method to create gound truth trajectory 
        return self.intended_next_state(s, a)

    def intended_next_state(self, state_in, action_in: int):
        # copy of step without actually moving to next state 
        
        assert self.action_space.contains(
            action_in
        ), f"{action_in!r} ({type(action_in)}) invalid"

        position, velocity = state_in
        velocity += (action_in - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        next_state = (position, velocity)
        if self.render_mode == "human":
            self.render()

        return np.array(next_state, dtype=np.float32)

    def P_sasp(self,s,a,sp):
        '''
        input args: 
        s: either current timestep partition index or state
        a: current discrete action
        sp: either next timestep partition index or state

        returns:
        transition probability

        '''

        sum_monte_carlo = 0
        size_Sg = 0
        num_samples = 0

        # check which input is an index and which one is the state
        # state will be two dimensional but index will be scalar 
        if len(s) == 2 and len(sp)== 1:
            # first term in Gibbs sampler product, integrate over second input
            samples_nxt_s,size_Sg = self.discrete_samples_to_estimate_integral(sp)
            num_samples = len(samples_nxt_s)
            # intended next state is same for all samples because current state and action are not changing 
            intended_nxt_s = self.intended_next_state(s, a)
            for st in samples_nxt_s:
                # indicator function checking if the sampled next state is intended next state 
                if st == intended_nxt_s:
                    sum_monte_carlo += 1
            
        elif len(s) == 1 and len(sp)== 2:
            # third term in Gibbs sampler product, integrate over first input
            samples_currnt_s,size_Sg = self.discrete_samples_to_estimate_integral(s)
            num_samples = len(samples_currnt_s)
            for st in samples_currnt_s:
                # intended next state varies because current state varies 
                intended_nxt_s = self.intended_next_state(st, a)
                # indicator function checking if the input next state is intended next state for (sampled current state, action) 
                if sp == intended_nxt_s:
                    sum_monte_carlo += 1

        else:
            raise ValueError("invalid input to P_sasp in DiscretizedStateMountainCarEnv")

        # return (size of state space) * sum/(number of samples) as monte carlo approximation
        return (size_Sg * sum_monte_carlo * 1/(num_samples))
        
    def insertNoise(self, s, a):
        '''
        input args: 
        s: a continuous state
        a: a discrete action

        this method increases position part of current state by 10% of (position span)/D in 
        the direction of action (left for action 0, right for action 1), with prob self.insertNoiseprob

        returns:
        (noised continuous state, noised action)

        '''
        if random.uniform(0.0, 1.0) < self.insertNoiseprob:
            delta = (self.max_position -self.min_position)*0.01*self.percChangeInPos/self._D 
            delta_speed = (self.max_speed * 2)*0.01*self.percChangeInPos/self._D 
            if a == 0 or a == 1: # current acceleration to left or no acceleration 
                a = 2 
                # pull car to left 
                s[0] -= delta 
                s[1] -= delta_speed
            else: # current accelaration to right 
                a = 0 
                # pull car to right 
                s[0] += delta 
                s[1] += delta_speed

        return (s,a) 

    def obs_model(self, sg, ag, so, ao): 
        '''
        input args: 
        sg: index for ground truth state partition 
        ag: ground truth action
        so: observed continuous state
        ao: observed action

        returns: 
        probability of observing the input

        '''

        if ag == ao: 
            # action do not have noise insertion, so only state should vary 
            delta = (self.max_position -self.min_position)*0.1/self._D  
            #  approximate integral over sg 
            sum_monte_carlo = 0
            samples_sg,size_Sg = self.discrete_samples_to_estimate_integral(sg)
            num_samples = len(samples_sg)
            for sg_cont in samples_sg:
                if ((ag == 0 or ag == 1) and (so == sg_cont - delta)) or \
                    ((ag == 2) and (so == sg_cont + delta)):
                    # state have noise
                    sum_monte_carlo += self.insertNoiseprob
                else: 
                    # noise free 
                    sum_monte_carlo += 1- self.insertNoiseprob

            return (size_Sg * sum_monte_carlo * 1/(num_samples))

        return 0.0

    def state_space_partitions(self):
        # returns the list of partition spaces 
        return self._obs_sp_partitions

    def state_space_size(self):
        # return the size of overall state space
        # low_array = self.observation_space.low 
        low_array = np.array([self.min_position, -self.max_speed])
        # high_array = self.observation_space.high 
        high_array = np.array([self.max_position, self.max_speed])
        return np.prod(high_array-low_array)