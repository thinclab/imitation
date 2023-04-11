from gym.envs.classic_control.mountain_car import MountainCarEnv
from numpy import *
from gym import Env, spaces
import numpy as np
import math 
import collections
import concurrent.futures
import sys
import time

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
    def __init__(self, D=10, n_smp_pr_dim=5):
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
            low_array = self._obs_sp_partitions[ind].low
            high_array = self._obs_sp_partitions[ind].high
            list_states = np.random.uniform(low_array,high_array,(self._n_smp_pr_dim**self.num_dims,self.num_dims))

        return list_states,np.prod(high_array-low_array)

    def step_sa(self, s, a): # AdversarialTrainer > train_disc uses this method to create gound truth trajectory 
        return self.intended_next_state(s, a)

    def intended_next_state(self, state_in, action_in: int):
        # copy of step without actually moving to next state 
        
        position, velocity = state_in
        velocity += (action_in - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        next_state = (position, velocity)

        return np.array(next_state, dtype=np.float32)

    def P_sasp(self,s,a,sp):
        '''
        input args: 
        s: current timestep state
        a: current discrete action
        sp: next timestep partition index

        returns:
        transition probability

        '''

        sum_monte_carlo = 0
        size_Sg = 0
        num_samples = 0

        # P_s_prevs_preva term in Gibbs sampler product on overleaf, integrate over second input
        samples_nxt_s,size_Sg = self.discrete_samples_to_estimate_integral(sp)
        num_samples = len(samples_nxt_s)
        # intended next state is same for all samples because current state and action are not changing 
        intended_nxt_s = self.intended_next_state(s, a)
        for nxt_st in samples_nxt_s:
            # indicator function checking if the sampled next state is intended next state 
            if all(nxt_st == intended_nxt_s):
                sum_monte_carlo += 1

        '''
        # check which input is an index and which one is the state
        # state will be two dimensional but index will be scalar 

        if isinstance(s, (collections.abc.Sequence, np.ndarray)) and not isinstance(sp, (collections.abc.Sequence, np.ndarray)):
            # P_s_prevs_preva term in Gibbs sampler product, integrate over second input
            samples_nxt_s,size_Sg = self.discrete_samples_to_estimate_integral(sp)
            num_samples = len(samples_nxt_s)
            # intended next state is same for all samples because current state and action are not changing 
            intended_nxt_s = self.intended_next_state(s, a)
            for nxt_st in samples_nxt_s:
                # indicator function checking if the sampled next state is intended next state 
                if all(nxt_st == intended_nxt_s):
                    sum_monte_carlo += 1
            
        elif not isinstance(s, (collections.abc.Sequence, np.ndarray)) and isinstance(sp, (collections.abc.Sequence, np.ndarray)):
            # P_nexts_s_a term in Gibbs sampler product, integrate over first input
            samples_currnt_s,size_Sg = self.discrete_samples_to_estimate_integral(s)
            num_samples = len(samples_currnt_s)
            for st in samples_currnt_s:
                # intended next state varies because current state varies 
                intended_nxt_s = self.intended_next_state(st, a)
                # indicator function checking if the input next state is intended next state for (sampled current state, action) 
                if all(sp == intended_nxt_s):
                    sum_monte_carlo += 1

        else:
            raise ValueError("invalid input to P_sasp in DiscretizedStateMountainCarEnv")
        '''

        # return (size of state space) * sum/(number of samples) as monte carlo approximation
        return_v = (size_Sg * sum_monte_carlo * 1/(num_samples))
        assert (return_v >= 0 and return_v <= 1),"P_sasp returning invalid prob value {} ".format(return_v)
        return return_v

    def P_sasp2(self,s,a,sp):
        '''
        input args: 
        s: current timestep partition index 
        a: current discrete action
        sp: next timestep state

        returns:
        transition probability

        '''

        sum_monte_carlo = 0
        size_Sg = 0
        num_samples = 0

        # P_nexts_s_a term in Gibbs sampler product, integrate over first input
        samples_currnt_s,size_Sg = self.discrete_samples_to_estimate_integral(s)
        num_samples = len(samples_currnt_s)
        for st in samples_currnt_s:
            # intended next state varies because current state varies 
            intended_nxt_s = self.intended_next_state(st, a)
            # indicator function checking if the input next state is intended next state for (sampled current state, action) 
            if all(sp == intended_nxt_s):
                sum_monte_carlo += 1
        
        return_v = (size_Sg * sum_monte_carlo * 1/(num_samples))
        assert (return_v >= 0 and return_v <= 1),"P_sasp2 returning invalid prob value {} ".format(return_v)
        return return_v

    def P_sasp2_no_intgrl(self,s,a,sp):
        '''
        input args: 
        s: current timestep partition index 
        a: current discrete action
        sp: next timestep state

        For both ends of input partition s, we can compute intended next state with action a. 
        If nexts state sp falls in between these two intended next states, then 
        prob of sp being next state is 1 else 0. 

        returns:
        transition probability

        '''
        start_P_sasp2_no_intgrl = time.time()

        low_array = self._obs_sp_partitions[s].low
        high_array = self._obs_sp_partitions[s].high
        beg_partition = low_array
        end_partition = high_array

        beg_intended_nxt_s = self.intended_next_state(beg_partition, a)
        end_intended_nxt_s = self.intended_next_state(end_partition, a)

        return_v = 0
        if all((sp >= beg_intended_nxt_s) & (sp <= end_intended_nxt_s)):
            # next state is in between intended states of partition edges 
            return_v = 1
        else:
            return_v = 0

        # print("time taken in P_sasp2_no_intgrl method {} ".format((time.time()-start_P_sasp2_no_intgrl)/60))
        return return_v

    def insertNoise(self, s, a):
        '''
        input args: 
        s: a continuous state
        a: a discrete action

        this method increases position+speed of current state by some percent of (position span)/D in 
        the direction of action (left for action 0, right for action 1), 
        and reverses action, 
        and with prob self.insertNoiseprob

        returns:
        (noised continuous state, noised action)

        '''
        if random.uniform(0.0, 1.0) < self.insertNoiseprob:
            delta = (self.max_position -self.min_position)*0.01*self.percChangeInPos/self._D 
            delta_speed = (self.max_speed * 2)*0.01*self.percChangeInPos/self._D 
            # if a == 0 or a == 1: # current acceleration to left or no acceleration 
                # a = 2 # pull car to right 
                # print('a == 0 inserted a = 1 # dont accelerate')
                
                # s[0] -= delta 
                # s[1] -= delta_speed
            # else: # current accelaration to right 
            #     a = 0 # pull car to left
                # print('a == 2 inserted a = 1 # dont accelerate')                 
                # s[0] += delta 
                # s[1] += delta_speed
            if a == 0:
                a = 1 # don't accelerate
                s[0] = 0
                s[1] = 0
            elif a == 2: # current accelaration to right 
                a = 1 # don't accelerate
                s[0] = 0
                s[1] = 0

        return (s,a) 

    def obs_model(self, sg, ag, so, ao): 
        '''
        input args: 
        sg_ind: index for ground truth state partition 
        ag: ground truth action
        so: observed continuous state
        ao: observed action

        does noise free version of observed state falls in state partition? 
        if yes, then chances of observed input state-action is same as (prob of inserting noise) 
        else 1-(prob of inserting noise)  

        returns: 
        probability of observing the input

        '''

        sg_ind = sg
        delta = (self.max_position - self.min_position)*0.01*self.percChangeInPos/self._D 
        delta_speed = (self.max_speed * 2)*0.01*self.percChangeInPos/self._D 
        noise_free_so = [sys.maxsize, sys.maxsize]
        low_sg = self._obs_sp_partitions[sg_ind].low
        high_sg = self._obs_sp_partitions[sg_ind].high

        if ag == 0: #(ag == 0 or ag == 1):
            if ao == 1: #(ao ==2): 
                noise_free_so[0] = so[0] + delta 
                noise_free_so[1] = so[0] + delta_speed 

                if all((noise_free_so >= low_sg) & (noise_free_so <= high_sg)):
                    return self.insertNoiseprob
        elif ag == 2:
            if ao == 1: #(ao ==0): 
                noise_free_so[0] = so[0] - delta 
                noise_free_so[1] = so[0] - delta_speed 

                if all((noise_free_so >= low_sg) & (noise_free_so <= high_sg)):
                    # state have noise
                    return self.insertNoiseprob
        
        # if no noise, then return remaining prob mass
        return 1 - self.insertNoiseprob

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