from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np
from gym import spaces


class HalfCheetahEnvMdfdWeights(HalfCheetahEnv):
    def __init__(self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.09,
        cov_diag_val_transition_model = 0.0001, 
        cov_diag_val_st_noise = 0.1,
        cov_diag_val_act_noise = 0.1, 
        noise_insertion=False):

        self.num_samples = 100000
        self.obs_size = 17
        self.act_size = 6
        self.transition_model_cov = np.diag(np.repeat(cov_diag_val_transition_model,self.obs_size))
        self.cov_diag_val_st_noise = cov_diag_val_st_noise
        self.cov_diag_val_act_noise = cov_diag_val_act_noise
        self.noise_insertion = noise_insertion
        super(HalfCheetahEnvMdfdWeights, self).__init__(\
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale)

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

    def insertNoise(self, s, a):
        sg_t, ag_t = s, a
        means_covs = self.mean_cov_sao_t_gvn_sag_t(sg_t, ag_t)
        so_t = np.random.multivariate_normal(means_covs[0], means_covs[1], (1))[0]
        ao_t = np.random.multivariate_normal(means_covs[2], means_covs[3], (1))[0]
        
        return so_t, ao_t
    
    def step(self, action):
        if not self.noise_insertion:
            x_position_before = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            x_position_after = self.sim.data.qpos[0]
            x_velocity = (x_position_after - x_position_before) / self.dt

            ctrl_cost = self.control_cost(action)

            forward_reward = self._forward_reward_weight * x_velocity

            observation = self._get_obs()
            reward = forward_reward - ctrl_cost
            done = False
            info = {
                "x_position": x_position_after,
                "x_velocity": x_velocity,
                "reward_run": forward_reward,
                "reward_ctrl": -ctrl_cost,
            }
        else:
            # noise insertion is needed for saving state action pairs 
            # without reward value because learner can't see expert's reward 
            reward = 0.0
            done = False 
            info = {}
            
            observation = self._get_obs()
            sg_t, ag_t = observation, action
            means_covs = self.mean_cov_sao_t_gvn_sag_t(sg_t, ag_t)           
            so_t = np.random.multivariate_normal(means_covs[0], means_covs[1], (1))[0]
            observation = so_t

            # setting noised state in simulation
            x_position_before = self.sim.data.qpos[0]
            full_state = np.append([x_position_before],observation[:self.obs_size])
            qpos = np.array(full_state[:9])
            qvel = np.array(full_state[9:self.obs_size+1])
            self.set_state(qpos, qvel)

            ao_t = np.random.multivariate_normal(means_covs[2], means_covs[3], (1))[0]
            self.do_simulation(ao_t, self.frame_skip)

        return observation, reward, done, info
