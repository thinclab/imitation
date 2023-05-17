from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np
from gym import spaces


class HalfCheetahEnvMdfdWeights(HalfCheetahEnv):
    def __init__(self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.09):
        super(HalfCheetahEnvMdfdWeights, self).__init__(\
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale)
        self.num_samples = 10000
        self.obs_size = 17
        self.act_size = 6

    def state_samples_to_estimate_LBA(self):
        # return sampled states to be used for LBA computation
        # as episode finishes when first dimension crosses [0.2,1.0] 
        # it's better to get samples from that window
        low = self.observation_space.low
        low[:] = -np.iinfo('uint16').max
        high = self.observation_space.high
        high[:] = np.iinfo('uint16').max
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        obspace = spaces.Box(low, high, dtype=observation.dtype)

        # st_tm = time.time()
        sampled_states = []
        for i in range(self.num_samples):
            sampled_states.append(obspace.sample())

        # print("time taken to sample {} states: {} min".format(self.num_samples,(time.time()-st_tm)/60)) 

        max_diff_acts = self.action_space.high - self.action_space.low
        return sampled_states, max_diff_acts