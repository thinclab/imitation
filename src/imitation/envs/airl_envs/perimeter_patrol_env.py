import numpy as np
# import numpy
import sys
from numpy import *

from gym import Env, spaces
from gym.utils import seeding
import random

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.state = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.state)

    def step(self, a):
        transitions = self.P[self.state][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.state = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

class PatrolModel(DiscreteEnv):

    """
    Environment for Pick Inspect Place behavior for sorting onions 
    """

    def __init__(self):

        p_fail = 0.05 
        terminal=PatrolState(np.array( [-1,-1, -1] )) 

        self._actions = [PatrolActionMoveForward(), PatrolActionTurnLeft(), 
        PatrolActionTurnRight(), PatrolActionStop()] 
        self._actionList = ['PatrolActionMoveForward', 'PatrolActionTurnLeft', 
        'PatrolActionTurnRight', 'PatrolActionStop'] 
        self._p_fail = float(p_fail) 
        self._terminal = terminal 
        
        """All states in the MDP"""
        self._stateList = []
        self._map = np.array( [[0, 1, 1, 1, 1, 1, 1, 1, 1], 
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 1]])
        nz = np.nonzero(self._map == 1)
        for o in range(4):
            for c, ind in enumerate(nz[0]):
                s = array( [nz[0][c], nz[1][c], o] )
                self._stateList.append( PatrolState( s ) )

        nS = len(self._stateList) 
        nA = len(self._actions) 
        self.isd = np.zeros(nS)
        for s in range(nS):
            self.isd[s] = 1.0/nS

        print ("\n total number of states: "+str(nS)+"\n")

        self._rewardmodel = Boyd2RewardGroupedFeatures(self) 
        weights = [1, 0, 0, 0, 0.75, 0] 
        # weights = [1, -0.75, 0.0, -0.75, 0.75, -0.75] 
        norm_weights = [float(i)/sum(np.absolute(weights)) for i in weights] 
        self._rewardmodel.params = norm_weights #[0.5714285714285714, 0.0, 0.0, 0.0, 0.42857142857142855, 0.0]

        P = {}
        p_str = ""
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}
            for a in range(nA):
                pstate = self._stateList[s]
                paction = self._actions[a]
                p_str += "\n"+str((pstate,paction))+"\n"
                trans_dict = self.T(pstate,paction)
                rew = self._rewardmodel.reward(pstate,paction)
                list_p_sa = []
                for next_pstate in trans_dict:
                    ns = self._stateList.index(next_pstate) 
                    list_p_sa.append((trans_dict[next_pstate], ns, rew, False)) 
                    p_str += str((next_pstate,trans_dict[next_pstate]))+", " 
                P[s][a] = list_p_sa 
        
        self.P = P
        self._timestep = 0
        # self._episode_length = 40
        self._episode_length = 5
        # self._episode_length = 120

        # read expert's policy in one format and write in expected format
        # p_dict = {}
        # policy_list = np.zeros(nS)
        # reader = open("./expected_policy_patrolling.txt", "r") 
        # for stateaction in reader: 
        #     temp = stateaction.split(" = ")
        #     if len(temp) < 2: continue
        #     state = temp[0]
        #     action = temp[1].strip()
                                                    
        #     state = state[1 : len(state) - 1]
        #     pieces = state.split(",")	
        #     a = ""
        #     ps = PatrolState(np.array([int(pieces[0]), int(pieces[1]), int(pieces[2])]))
        #     ps_index = self._stateList.index(ps)
        #     # print("read action ",action)
        #     if action == "MoveForwardAction":
        #         a = 'PatrolActionMoveForward'
        #     elif action == "TurnLeftAction":
        #         a = 'PatrolActionTurnLeft'
        #     elif action == "TurnRightAction":
        #         a = 'PatrolActionTurnRight'
        #     else:
        #         a = 'PatrolActionStop'
            
        #     # print("read a ",a)
        #     p_dict[ps_index] = self._actionList.index(a)

        # with open('./expert_policy.txt', 'w') as writer:
        #     for ps_index in range(nS):
        #         writer.write(str(p_dict[ps_index]))
        #         writer.write("\n")
        
        # with open('./expert_policy_string.txt', 'w') as writer:
        #     for ps_index in range(nS):
        #         writer.write("{} = {}".format(self._stateList[ps_index],self._actionList[p_dict[ps_index]]))
        #         writer.write("\n")

        # exit(0)
        self.nS = nS

        self.insertNoiseprob = 0.05
        # self.insertNoiseprob = 0.1
        # self.insertNoiseprob = 0.3
        # self.insertNoiseprob = 0.6
        # self.insertNoiseprob = 0.9
        # self.insertNoiseprob = 0.95
        # self.insertNoiseprob = 0.99
        # self.insertNoiseprob = 0.998
        # self.insertNoiseprob = 0.999
        # self.insertNoiseprob = 1
        super(PatrolModel, self).__init__(nS, nA, self.P, self.isd) 

    def get_statelist(self):
        return self._stateList

    def get_actionlist(self):
        return self._actions

    def reset(self):
        self.state = categorical_sample(self.isd, self.np_random)
        # print("state sampled in reset ",self._stateList[self.state])
        self.lastaction = None
        self._timestep = 0
        # print("self.state ",self.state)
        return int(self.state)

    def bestpolicy_action(self, s):
        # function to return best possible expert action in a state
        # self._actions = [PatrolActionMoveForward(), PatrolActionTurnLeft(), 
        # PatrolActionTurnRight(), PatrolActionStop()] 

        best_a = None
        state_loc = self._stateList[s].location

        pstate = self._stateList[s]
        moveforwardobj = PatrolActionMoveForward()
        
        reward_turnright = self._rewardmodel.reward(pstate,PatrolActionTurnRight())
        reward_forward = self._rewardmodel.reward(pstate,PatrolActionMoveForward())
        
        if reward_turnright>0 and state_loc[2] == 2: # states right after turning around , facing away from end points
            best_a = self._actions.index(PatrolActionMoveForward())
        elif reward_turnright>0: # states while turning
            best_a = self._actions.index(PatrolActionTurnRight())
        elif reward_forward>0:  # is there no wall in front?
            best_a = self._actions.index(PatrolActionMoveForward())
        elif not self.is_legal(moveforwardobj.apply(self._stateList[s])):
            # can't move forward, facing walls
            # If it is at corner, do turn action based on location and orientation 
            if all(state_loc == [0,1,1]) or all(state_loc == [16,1,2]): 
                best_a = self._actions.index(PatrolActionTurnRight())
            elif all(state_loc == [0,1,2]) or all(state_loc == [16,1,3]):
                best_a = self._actions.index(PatrolActionTurnLeft())
            else: # all other states with agent facing walls
                best_a = self._actions.index(PatrolActionTurnRight())
        else:
            # if neither getting reward at turn nor on move forward nor facing walls, move forward
            best_a = self._actions.index(PatrolActionMoveForward())
        
        return best_a
    
    def get_expert_det_policy_list(self):
        policy_acts_expert = [self._actions.index(PatrolActionMoveForward())]*len(self._stateList)
        str_sa = ""
        for s_i in range(len(self._stateList)):
            policy_acts_expert[s_i] = self.bestpolicy_action(s_i)
            
            str_sa += "\n{} {}".format(self._stateList[s_i],self._actions[policy_acts_expert[s_i]])
        # print(str_sa)
        return policy_acts_expert

    def step_sa(self, s, a):
        current_s_backup = self.state
        self.state = s
        result = self.step(a)  
        self.state = current_s_backup
        return result[0]

    def step(self, a):
        self._timestep += 1
        transitions = self.P[self.state][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        # print("\nstep  s {}, a {}, ns {}, options {}".format\
        #     (self._stateList[self.state],self._actionList[a],self._stateList[s],transitions))
        # with open('/home/katy/imitation/rollout_from_policylist.txt', 'a') as writer:
        #     writer.write("\n step input {} self._timestep {} ".format(self._actions[a],self._timestep))
        
        self.state = s
        self.lastaction = a
        if self._timestep == self._episode_length:
            d = True
        
        # if self._timestep == 5:
        #     print("debug")

        return (int(s), r, d, {"prob": p})
    
    def T(self,state,action):
        """Returns a function state -> [0,1] for probability of next state
        given currently in state performing action"""
        result = NumMap()
        valid_actions = self.A(state)
        s_p = action.apply(state)
        if action not in valid_actions or not self.is_legal(s_p) or s_p.__eq__(state):
            result[state] = 1
        else:
            result[state] = self._p_fail
            result[s_p] = 1-self._p_fail
        
        return result 

    def P_sasp(self,s,a,sp):
        '''
        input args: 
        s: current state
        a: current action
        sp: next state

        returns:
        transition probability

        '''
        res = None
        for (p,ns,rw,dn) in self.P[s][a]:
            if ns == sp:
                res = p
                break
        if not res:
            res = 0.0
        return res 

    def obs_model(self, sg, ag, so, ao): 
        '''
        input args: 
        sg: ground truth state
        ag: ground truth action

        returns: 
        probability of observing the input

        '''
        state_arr = self._stateList[sg].location
        act = self._actionList[ag]
        if act == 'PatrolActionMoveForward' and (state_arr[0]==0 or state_arr[1]==1):
            if so == sg and ao == self._actionList.index('PatrolActionTurnLeft'):
                return self.insertNoiseprob
            elif so == sg and ao == ag:
                return 1-self.insertNoiseprob
            else:
                return 0.0
        else:
            if so == sg and ao == ag:
                return 1.0
            else:
                return 0.0

    def insertNoise(self, s, a):
        '''
        input args: 
        s: current state
        a: current action
        returns:
        (noised state, noised action)

        '''
        
        state_arr = self._stateList[s].location
        act = self._actionList[a]

        a_new = a
        if act == 'PatrolActionMoveForward':
            if random.uniform(0.0, 1.0) < self.insertNoiseprob:
                a_new = self._actionList.index('PatrolActionTurnLeft')

        # if act == 'PatrolActionTurnLeft' and (state_arr[0]==0 or state_arr[1]==0):
        if act == 'PatrolActionTurnRight':
            if random.uniform(0.0, 1.0) < self.insertNoiseprob:
                a_new = self._actionList.index('PatrolActionMoveForward')

        return (s,a_new)
    
    
    def get_obstr_actstr(self,s,a):
        obstr = str(self._stateList[s])
        actstr = self._actionList[a]
        return (obstr,actstr)
    
    
    def S(self):
        return list(range(self.nS))

    def A(self,state=None):
        """All actions in the MDP is state=None, otherwise actions available
        from state"""
                
        return self._actions

    def is_terminal(self, state):
        '''returns whether or not a state is terminal'''
        return all(state == self._terminal)
    
    def is_legal(self,state):
        loc = state.location
        (r,c) = self._map.shape
        
        return loc[0] >= 0 and loc[0] < r and \
            loc[1] >= 0 and loc[1] < c and \
            loc[2] >= 0 and loc[2] < 4 and \
            self._map[ loc[0],loc[1] ] == 1
    
    def __str__(self):
        format = 'GWModel [p_fail={},terminal={}]'
        return format.format(self._p_fail, self._terminal)
    
    def info(self):
        result = [str(self) + '\n']
        map_size = self._map.shape
        for i in reversed(range(map_size[0])):
            for j in range(map_size[1]):
                if self._map[i,j] == 1:
                    result.append( '[O]' )
                else:
                    result.append( '[X]')
            result.append( '\n' )
        return ''.join(result)


class State(object):
    '''
    State of an MDP
    '''
    pass

class PatrolState(State):
    
    def __init__(self, location=array( [0,0,0] )):
        self._extension = 0
        self._location = location
    
    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, loc):
        self._location = loc
        
    def __str__(self):
        return 'PatrolState: [location={}]'.format(self.location)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        try:
            return all( self.location == other.location) # epsilon error
        except Exception:
            return False
    
    def __hash__(self):
        loc = self.location # hash codes for numpy.array not consistent?
        return (loc[0], loc[1], loc[2]).__hash__()
    
    def conflicts(self, otherstate):
        return self.location[0] == otherstate.location[0] and self.location[1] == otherstate.location[1]


class Action(object):
    '''
    Action in an MDP
    '''
    pass

# Action classes

class PatrolActionMoveForward(Action):
    
    def apply(self,gwstate):
        
        if gwstate.location[2] == 0:
            return PatrolState( gwstate.location + array( [0,1,0] ) )
        if gwstate.location[2] == 1:
            return PatrolState( gwstate.location + array( [-1,0,0] ) )
        if gwstate.location[2] == 2:
            return PatrolState( gwstate.location + array( [0,-1,0] ) )
        if gwstate.location[2] == 3:
            return PatrolState( gwstate.location + array( [1,0,0] ) )
            
    def __str__(self):
        return "PatrolActionMoveForward"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "PatrolActionMoveForward"
        except Exception:
            return False

    def __hash__(self):
        return 0
           

class PatrolActionStop(Action):
    
    def apply(self,gwstate):
        return PatrolState( gwstate.location )
    
    def __str__(self):
        return "PatrolActionStop"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "PatrolActionStop"
        except Exception:
            return False
           
    def __hash__(self):
        return 1
           
class PatrolActionTurnLeft(Action):
    
    def apply(self,gwstate):
        
        next_loc = gwstate.location + array( [0,0,1] )
        if next_loc[2] > 3:
            next_loc[2] = 0
        
        returnval = PatrolState( next_loc )
        return returnval
    
    def __str__(self):
        return "PatrolActionTurnLeft"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "PatrolActionTurnLeft"
        except Exception:
            return False

    def __hash__(self):
        return 2


class PatrolActionTurnRight(Action):
    
    def apply(self,gwstate):

        next_loc = gwstate.location + array( [0,0,-1] )
        if next_loc[2] < 0:
            next_loc[2] = 3
            
        returnval = PatrolState( next_loc )
        return returnval
    
    def __str__(self):
        return "PatrolActionTurnRight"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "PatrolActionTurnRight"
        except Exception:
            return False
    
    def __hash__(self):
        return 4


class Reward(object):
    '''
    A Reward function stub
    '''

    def __init__(self):
        self._params = []
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self,_params):
        self._params = _params
        
    def reward(self, state, action):
        raise NotImplementedError()
    
class LinearReward(Reward):
    '''
    A Linear Reward function stub
    
    params: weight vector equivalent to self.dim()
    '''
    def __init__(self,dim):
        super(LinearReward,self).__init__()

    def features(self, state, action):
        raise NotImplementedError()
    
    @property
    def dim(self):
        raise NotImplementedError()
    
    def reward(self, state, action):
        return float(dot(self.params, self.features(state,action)))

class Boyd2RewardGroupedFeatures(LinearReward):
    def __init__(self, model):
        self._dim = 6
        self._model = model
        super(Boyd2RewardGroupedFeatures,self).__init__(self._dim)
        
    @property
    def dim(self):
        return self._dim

    def features(self, state, action):  
        result = np.zeros( self._dim )
        next_state = action.apply(state)
        moved = False

        if self._model.is_legal(next_state) and not all(next_state.location[0:2] == state.location[0:2]):
            moved = True
            result[0] = 1
        
        if (self._model.is_legal(next_state) and not moved):
            if action.__class__.__name__ == "PatrolActionTurnLeft" or action.__class__.__name__ == "PatrolActionTurnRight":

                if (state.location[0] >= 1 and state.location[0] <= 15):# longer hallway
                    result[1] = 1
                if ((state.location[0] <= 1 or state.location[0] >= 15) and state.location[1] <= 2): # corners
                    result[2] = 1
                if (state.location[1] >= 2 and state.location[1] <= 3): # small hallway
                    result[3] = 1
                if (state.location[1] >= 4 and state.location[1] <= 5): # small hallway
                    result[4] = 1
                if (state.location[1] >= 6 and state.location[1] <= 8): # dead end
                    result[5] = 1

        return result
    
    def __str__(self):
        return 'Boyd2RewardGroupedFeatures'
        
    def info(self, model = None):
        result = 'Boyd2RewardGroupedFeatures:\n'
        if model is not None:        
            for a in self._actions:
                result += str(a) + '\n'
                for s in model.S():
                     result += '|{: 4.4f}|'.format(self.reward(s, a))
                result += '\n\n'
        return result

class NumMap(dict):
    """A dict that explicitly maps to numbers"""
    
    def max(self):
        return self[self.argmax()]
    
    def min(self):
        return self[self.argmin()]
    
    def argmax(self):
        if len(self) == 0:
            raise Exception('Cannot take argmax/min without any choices')
        return max( [(value,key) for (key,value) in self.iteritems()] )[1]
    
    def argmin(self):
        if len(self) == 0:
            raise Exception('Cannot take argmax/min without any choices')
        return min( [(value,key) for (key,value) in self.iteritems()] )[1]
    
    def normalize(self):
        for val in self.values():
            if val < 0:
                raise Exception('Cannot normalize if numbers < 0')
        
        sumvals = sum(self.values())
        result = NumMap()
        for (key,val) in self.iteritems():
            result[key] = float(val)/sumvals
        return result
    
    def __getitem__(self,key):
        if not key in self:
            return 0.0
        return super(NumMap,self).__getitem__(key)
    
    def __str__(self):
        result_list = ["{"]
        for (k,v) in self.items():
            result_list.append( '{}:{:4.4f}, '.format(str(k), v) )
        result_list[-1] = result_list[-1][:-2]
        result_list.append('}')
        return ''.join(result_list)
    
    def __eq__(self, other):
        try:
            for (key,val) in self.items():
                if val != other[key]:
                    return False
            for (key, val) in other.items():
                if val != self[key]:
                    return False
            return True
        except Exception:
            return False
    
    def info(self):
        result = ['NumMap:\n']
        for (k,v) in self.items():
            result.append( '\t{} ===== {:4.4f}\n'.format(str(k), v))
        return ''.join(result)
