import numpy as np
import sys
from numpy import dot

from gym import Env, spaces
from gym.utils import seeding

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
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

class SortingOnions(DiscreteEnv):

    """
    Environment for Pick Inspect Place behavior for sorting onions 
    """

    def __init__(self):

        p_fail = 0.05 
        params_pickinspectplace = [2.5, 2, 2, 2.5, 1, -0.5, -0.5, 2, -0.5, -0.5, 2] # [2,1,2,1,0.2,0.1,0,4,0,0,4] 
        # params_pickinspectplace = [2.5, 2, 2, 2.5, 1, 2, -0.5, 2, -0.5, -0.5, 2]
        params = params_pickinspectplace 
        norm_weights_reward7 = [float(i)/sum(np.absolute(params)) for i in params] 
        reward_obj = sortingReward7(len(norm_weights_reward7)) 
        reward_obj.params = norm_weights_reward7

        self.statelist = []
        for ol in [0,1,2,3]:
            for pr in [0,1,2]:
                for el in [0,1,2,3]:
                    # onion can be either on conveyor or
                    # not on conveyor with its location same as gripper 
                    if (ol!=0 and ol==el) or ol==0:
                        for ls in [2]:
                            self.statelist.append([ol, pr, el, ls]) 
        self.statelist.append([-1,-1,-1,-1]) 

        # self.statelist = [[ 1, 0, 1, 2],\
        #     [ 2, 2, 2, 2],\
        #     [ 0, 2, 2, 2],\
        #     [ 3, 2, 3, 2],\
        #     [ 1, 1, 1, 2],\
        #     [ 4, 2, 0, 2],\
        #     [ 0, 0, 0, 1],\
        #     [ 3, 0, 3, 1],\
        #     [ 2, 2, 2, 1],\
        #     [ 0, 0, 2, 1],\
        #     [ 2, 2, 2, 0],\
        #     [0, 2, 0, 2],\
        #     [0, 2, 2, 0],\
        #     [0,1,0,0],[0,1,1,0],[0,1,2,0],[0,1,3,0],[0,2,1,0],[0,2,3,0],\
        #     [3,1,3,0],[0,0,1,1],[0,0,3,1],\
        #     [0,2,1,2],[0,2,3,2],\
        #     [-1,-1,-1,-1]] # sink state
            # ,[-2,-2,-2,-2]] # terminal state

        self.actionlist = [InspectAfterPicking(),InspectWithoutPicking(),\
        Pick(),PlaceOnConveyor(),PlaceInBin(),ClaimNewOnion(),\
        ClaimNextInList()]

        nS = len(self.statelist) 
        nA = len(self.actionlist) 
        P = {} 
        start_ss = sortingState(0, 2, 0, 2) 
        sink_ss = sortingState(-1, -1, -1, -1) 
        # term_ss = sortingState(-2, -2, -2, -2) 
        p_str = ""
        
        for s in range(len(self.statelist)):
            P[s] = {a: [] for a in range(nA)}
            ol, pr, el, ls = self.statelist[s][0], self.statelist[s][1], self.statelist[s][2], self.statelist[s][3]
            
            for a in range(len(self.actionlist)):
                # if source state is sink or terminal, then next state is too.
                # continue to next iteration.
                if  ol == -1:
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue

                # if  ol == -2:
                #     P[s][a] = [(1.0, s, 0.0, True)]
                #     continue

                ss = sortingState(ol, pr, el, ls)
                sa = self.actionlist[a] 
                new_s = None

                if sa in self.A(ss): 
                    nss = sa.apply(ss) 
                    new_s = self.statelist.index([nss._onion_location, nss._prediction, nss._EE_location, nss._listIDs_status])

                    if a==3 or a==4:
                        done = True
                    else:
                        done = False
                    
                    # nss = sa.apply(ss) 
                    # if (nss == term_ss):
                    #     # terminal state ends the episode
                    #     new_s = self.statelist.index([-2,-2,-2,-2])
                    #     done = True
                    # else: 
                    #     new_s = self.statelist.index([nss._onion_location, nss._prediction, nss._EE_location, nss._listIDs_status])
                    #     done = False
                    
                    rew = reward_obj.reward(ss, sa)

                    # if (nss == ss) or (nss == term_ss): 
                    # action leads to terminal state
                    if (nss == ss): 
                        # if intended next state is same as current state , the transition must be deterministic.
                        P[s][a] = [(1.0, new_s, rew, done)]
                    else:
                        P[s][a] = [(p_fail, s, rew, done),(1.0-p_fail, new_s, rew, done)]

                else: 
                    # if action is not allowed, it should lead to a sink state deterministically
                    new_s = self.statelist.index([-1,-1,-1,-1]) 
                    # there is should a penalty of reaching sink state, penalty higher than other weights
                    P[s][a] = [(1.0, new_s, -1.0, True)] 

                # if a==3 or a==4: 
                #     p_str += "\nP[s][a] for place actions"+str((s,a,P[s][a])) 

        self.isd = np.zeros(nS)
        start_ss_array = [0, 2, 0, 2]
        self.isd[self.statelist.index(start_ss_array)] = 1.0

        with open('./env_transition_dict.txt', 'w') as writer:
            writer.write(p_str)

        self.P = P

        super(SortingOnions, self).__init__(nS, nA, self.P, self.isd) 

    def get_statelist(self):
        return self.statelist

    def get_actionlist(self):
        return self.actionlist

    def get_actionlist_string(self):
        return np.array(['InspectAfterPicking','InspectWithoutPicking',\
        'Pick','PlaceOnConveyor','PlaceInBin','ClaimNewOnion',\
        'ClaimNextInList'])

    def sarray_ind_to_value(self):
        ol_map = ['conv','eye','bin','home','placedconv']
        pr_map = ['bad', 'good','unknown']
        el_map = ['conv','eye','bin','home'] 
        ls_map = ['empty','notempty','unavailable'] 
        r_args = [ol_map, pr_map, el_map, ls_map]
        return r_args 

    def perfect_demonstrator_det_policy_list(self):
        policy_acts_expert = [2]*len(self.statelist)
        
        for s_i in range(len(self.statelist)):
            ol, pr, el, ls = self.statelist[s_i]
            if ol == 0: 
                # onion on conveyor
                if pr == 2:
                    # pred unknown
                    policy_acts_expert[s_i] = 2 # pick 
                else: 
                    # known 
                    policy_acts_expert[s_i] = 5 # claim new 
            elif ol == 3:
                # onion at home (el has to be same by def of statelist) 
                if pr == 2:
                    # unknown 
                    policy_acts_expert[s_i] = 0 # inspect 
                elif pr == 0:
                    policy_acts_expert[s_i] = 4 # to bin 
                else:
                    policy_acts_expert[s_i] = 3 # to conv
            elif ol == 1:
                # onion at eye (by def of statelist, el has to be same)
                if pr == 2:
                    # unknown 
                    policy_acts_expert[s_i] = 0 # inspect 
                elif pr == 0:
                    policy_acts_expert[s_i] = 4 # to bin 
                else:
                    policy_acts_expert[s_i] = 3 # to conv                    
            else:
                # ol == 2, onion at bin 
                policy_acts_expert[s_i] = 5 # claim new, only action valid 
        
        return policy_acts_expert

    def A(self,state=None):
        ''' {
        0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 
        2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion', 5: 'InspectWithoutPicking', 
        6: 'ClaimNextInList'
        } '''

        if not state: 
            # res = [InspectWithoutPicking(), Pick(), ClaimNewOnion(), ClaimNextInList(), \
            # PlaceOnConveyor(), PlaceInBin(), InspectAfterPicking()] 
            res = [None, Pick(), ClaimNewOnion(), None, \
            PlaceOnConveyor(), PlaceInBin(), InspectAfterPicking()] 
            return res 
        
        res = [] 
        if state._onion_location == 0: 
            # on conveyor (not picked yet) 
            if state._listIDs_status == 2: 
                # list unavailable 
                res = [Pick(),ClaimNewOnion()] 
            # elif state._listIDs_status == 0: 
            #     # res = [InspectWithoutPicking()]
            #     res = [Pick(), ClaimNewOnion()]
            # else:
            #     # list not empty
            #     if state._prediction == 2:
            #         res = [ClaimNextInList()]
            #     else:
            #         res = [Pick()]

        elif state._onion_location == 1:
            if (state._listIDs_status == 2):
                if state._prediction == 0:
                    res = [PlaceInBin(),PlaceOnConveyor()]
                elif state._prediction == 1:
                    res = [PlaceOnConveyor(),PlaceInBin()]
                else: 
                    res = [InspectAfterPicking()]
            # else: 
            #     if state._prediction == 0:
            #         res = [PlaceInBin()]
            #     elif state._prediction == 1:
            #         res = [PlaceOnConveyor()]
            #     else: 
            #         res = [InspectAfterPicking()]

        elif state._onion_location == 2:
            if state._listIDs_status == 2: 
                res = [ClaimNewOnion()] 
            # elif state._listIDs_status == 0:  
            #     res = [Pick(), ClaimNewOnion()]
            # else:
            #     res = [ClaimNextInList()]

        elif state._onion_location == 3:
            if state._prediction == 2: 
                res = [InspectAfterPicking()] 
            elif state._prediction == 0: 
                res = [PlaceInBin()] 
            else: 
                res = [PlaceOnConveyor()] 

        else: 
            if state._listIDs_status == 2: 
                res = [ClaimNewOnion()] 
            # elif state._listIDs_status == 0: 
            #     res = [Pick(), ClaimNewOnion()] 
            # else: 
            #     res = [ClaimNextInList()] 

        return res 

class State(object):
    '''
    State of an MDP
    '''
    pass

class sortingState(State):

    def __init__(self, onion_location = 0, prediction = -1, EE_location = 1, listIDs_status = 1):
        '''
        onion_location: 
        on the conveyor, or 0
        in front of eye, or 1
        in bin or 2
        at home after begin picked or 3 (in superficial inspection, onion is picked and placed)
        placed_on_conveyor 4
        prediction:
        0 bad
        1 good
        2 unkonwn before inspection
        EE_location:
        conv 0
        inFront 1
        bin 2 
        at home 3
        listIDs_status: 
        2 list not available (because rolling hasn't happened for current set of onions)
        ''' 
        
        self._onion_location = onion_location
        self._prediction = prediction
        self._EE_location = EE_location
        self._listIDs_status = listIDs_status
        self._hash_array = [5,3,4,3]

    @property
    def onion_location(self):
        return self._location_chosen_onion

    @onion_location.setter
    def onion_location(self, loc):
        self._location_chosen_onion = loc

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def onion_location(self, pred):
        self._prediction = pred

    @property
    def EE_location(self):
        return self._prediction

    @EE_location.setter
    def EE_location(self, ee_loc):
        self._EE_location = ee_loc

    @property
    def listIDs_status(self):
        return self._listIDs_status

    @listIDs_status.setter
    def listIDs_status(self, listIDs_status):
        self._listIDs_status = listIDs_status

    def __str__(self):
        return 'State: [{}, {}, {}, {}]'.format(self._onion_location, self._prediction,\
         self._EE_location, self._listIDs_status)
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        try:
            return (self._onion_location == other._onion_location \
            and self._prediction == other._prediction\
            and other._EE_location == self._EE_location\
            and other._listIDs_status == self._listIDs_status) # epsilon error
        except Exception:
            print ("Exception in __eq__(self, other)")
            return False
    
    def __hash__(self):
        row_major = self._onion_location+self._hash_array[0]*(self._prediction+self._hash_array[1]*(self._EE_location+self._hash_array[2]*self._listIDs_status))
        return (row_major).__hash__()

class Action(object):
    '''
    Action in an MDP
    '''
    pass

class InspectAfterPicking(Action):
    
    def apply(self,state):
        if state._prediction == 2:
            # pp = 0.5
            # pred = np.random.choice([1,0],1,p=[1-pp,pp])[0]
            pred = np.random.choice([1,0],1)[0]
            return sortingState( 1, pred, 1, 2 )
        else:
            return sortingState( 1, state._prediction, 
            1, state._listIDs_status ) 

    def __str__(self):
        return "InspectAfterPicking"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "InspectAfterPicking"
        except Exception:
            return False
    
    def __hash__(self):
        return (8).__hash__() 
    
class InspectWithoutPicking(Action):  
    
    def apply(self,state): 
        #  can not apply this action if a list is already available
        global num_objects
        pp = 0.5
        pp = 1*0.95
        ls = np.random.choice([1,0],1,p=[pp,1-pp])[0] 
        if (ls == 0):
            pred = 2
        else:
            pred = 0
        return sortingState( 0, pred, state._EE_location, ls )
   
    def __str__(self):
        return "InspectWithoutPicking"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "InspectWithoutPicking"
        except Exception:
            return False
    
    def __hash__(self):
        return (13).__hash__()

class Pick(Action):
    
    def apply(self,state): 
        # onion picked and is at home-pose of sawyer
        return sortingState( 3, state._prediction, 3, state._listIDs_status )
    
    def __str__(self):
        return "Pick"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "Pick"
        except Exception:
            return False
    
    def __hash__(self):
        return (2).__hash__()

class PlaceOnConveyor(Action):
    
    def apply(self,state): 
        # return sortingState( -2, -2, -2, -2 )
        return sortingState( 0, 2, 0, 2 )
    
    def __str__(self):
        return "PlaceOnConveyor"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "PlaceOnConveyor"
        except Exception:
            return False
    
    def __hash__(self):
        return (24).__hash__()

class PlaceInBin(Action):
    
    def apply(self,state): 
        # return sortingState( -2, -2, -2, -2 )
        # # most of attempts won't make list empty if it is not already empty or unavailable
        # # if list is available and 50% of objects are bad, then 1 out of 6 attempts make 
        # # list empty
        # global num_objects
        # if state._listIDs_status == 1:
        #     pp = 0.5
        #     ls = np.random.choice([1,0],1,p=[pp,1-pp])[0] 
        #     return sortingState( 2, 2, 2, ls )
        # else:
        return sortingState( 2, 2, 2, state._listIDs_status )

    def __str__(self):
        return "PlaceInBin"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "PlaceInBin"
        except Exception:
            return False
    
    def __hash__(self):
        return (10).__hash__()

class ClaimNewOnion(Action):

    def apply(self,state):
        # on conv, unknown, 
        return sortingState( 0, 2, state._EE_location, 2 )
    
    def __str__(self):
        return "ClaimNewOnion"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "ClaimNewOnion"
        except Exception:
            return False
    
    def __hash__(self):
        return (12).__hash__()

class ClaimNextInList(Action):

    def apply(self,state):
        
        if state._listIDs_status == 1:
            # if list not empty, then 
            return sortingState( 0, 0, state._EE_location, 1 )
        else:
            # else make onion unknown and list not available
            return sortingState( 0, 2, state._EE_location, state._listIDs_status )
    
    def __str__(self):
        return "ClaimNextInList"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return other.__class__.__name__ == "ClaimNextInList"
        except Exception:
            return False
    
    def __hash__(self):
        return (14).__hash__()

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
    def features(self, state, action):
        raise NotImplementedError()
    
    @property
    def dim(self):
        raise NotImplementedError()
    
    def reward(self, state, action):
        return float(dot(self.params, self.features(state,action)))

class sortingReward7(LinearReward):

    def __init__(self,dim):
        super(sortingReward7,self).__init__()
        self._dim = dim
        
    @property
    def dim(self):
        return self._dim
    
    def features(self, state, action):        
        result = np.zeros( self._dim )
        next_state = action.apply(state)

        '''
        Feature functions:
        // good placed on belt
        // not placing bad on belt
        // not placing good in bin
        // bad placed in bin
        '''

        if state._prediction == 1 and next_state._onion_location == 0:
            result[0] = 1
        if state._prediction == 0 and next_state._onion_location != 0:
            result[1] = 1
        if state._prediction == 1 and next_state._onion_location != 2:
            result[2] = 1
        if state._prediction == 0 and next_state._onion_location == 2:
            result[3] = 1

        # not staying still 
        if not (state._onion_location == next_state._onion_location and\
        state._prediction == next_state._prediction and\
        state._EE_location == next_state._EE_location and\
        state._listIDs_status == next_state._listIDs_status): 
            result[4] = 1

        # claim new onion from belt 
        if (next_state._prediction == 2 and \
        next_state._onion_location == 0): 
            result[5] = 1 

        # create list 
        if state._listIDs_status == 0 and next_state._listIDs_status == 1: 
            result[6] = 1 

        # picking an onion with unknown prediction 
        if state._onion_location == 0 and state._prediction == 2 \
        and (next_state._prediction == 2 and next_state._EE_location == 3): 
            result[7] = 1

        # picking an onion with known pred - blemished  
        if state._onion_location == 0 and state._prediction == 0 \
        and (next_state._prediction == 0 and next_state._EE_location == 3): 
            result[8] = 1

        # empty list 
        if state._listIDs_status == 1 and next_state._listIDs_status == 0: 
            result[9] = 1 

        # inspect picked onion 
        if state._onion_location == 3 and state._prediction == 2 and\
        next_state._prediction != 2: 
            result[10] = 1 

        return result
    
    def __str__(self):
        return 'sortingReward7'
        
    def info(self, model = None):
        result = 'sortingReward7:\n'
        if model is not None:        
            for a in self._actions:
                result += str(a) + '\n'
                for s in model.S():
                     result += '|{: 4.4f}|'.format(self.reward(s, a))
                result += '\n\n'
        return result

