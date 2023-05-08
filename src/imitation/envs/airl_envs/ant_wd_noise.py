from gym.envs.mujoco.ant import AntEnv
import numpy as np
from gym import spaces
import time

class AntEnvWdNoise(AntEnv):
    def __init__(self,
                 cov_diag_val_transition_model = 0.0001, 
                 cov_diag_val_st_noise = 0.5,
                 cov_diag_val_act_noise = 0.1):
        super(AntEnvWdNoise, self).__init__()
        self.obs_size = 27
        self.act_size = 8
        self.transition_model_cov = np.diag(np.repeat(cov_diag_val_transition_model,self.obs_size))
        self.cov_diag_val_st_noise = cov_diag_val_st_noise
        self.cov_diag_val_act_noise = cov_diag_val_act_noise
        self.num_samples = 10000

    def state_samples_to_estimate_LBA(self):
        # return sampled states to be used for LBA computation
        # as episode finishes when first dimension crosses [0.2,1.0] 
        # it's better to get samples from that window
        low = self.observation_space.low
        low[0] = 0.2
        low[1:] = -np.iinfo('uint16').max
        high = self.observation_space.high
        high[0] = 1.0
        high[1:] = np.iinfo('uint16').max
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
    
    def intended_next_state(self, state, action):
        # make input state current state. 
        qpos = self.init_qpos[:2] + state[:13]
        qvel = state[13:27]
        self.set_state(qpos, qvel)
        # get next state
        result, _, _, _ = self.step(action)
        return result 
    
    def mean_cov_sao_t_gvn_sag_t(self, sg_t, ag_t):
        # mean vector and covariance matrix to be used 
        # for Gaussian sampling of noisy state given ground truth state 
        mean_so_t_gauss = sg_t
        cov_so_t_gauss = np.diag(np.repeat(self.cov_diag_val_st_noise,mean_so_t_gauss.size))

        # mean vector and covariance matrix to be used 
        # for Gaussian sampling of noisy state given ground truth state 
        mean_ao_t_gauss = ag_t
        cov_ao_t_gauss = np.diag(np.repeat(self.cov_diag_val_act_noise,mean_ao_t_gauss.size))

        return mean_so_t_gauss, cov_so_t_gauss, mean_ao_t_gauss, cov_ao_t_gauss

    def mean_cov_sg_t_gvn_sag_tmns1(self, sg_tmns1, ag_tmns1):
        ## mean cov for P(sg | sg_tmns1, ag_tmns1)  
        mean_sg_t_gauss = self.intended_next_state(sg_tmns1, ag_tmns1)
        cov_sg_t_gauss = self.transition_model_cov

        return mean_sg_t_gauss, cov_sg_t_gauss

    def insertNoise(self, s, a):
        sg_t, ag_t = s, a
        means_covs = self.mean_cov_sao_t_gvn_sag_t(sg_t, ag_t)
        so_t = np.random.multivariate_normal(means_covs[0], means_covs[1], (1))[0]
        ao_t = np.random.multivariate_normal(means_covs[2], means_covs[3], (1))[0]
        
        return so_t, ao_t
          

