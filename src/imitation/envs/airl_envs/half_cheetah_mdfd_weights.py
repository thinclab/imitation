from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
import numpy as np
from gym import spaces
from numpy.linalg import inv

import time

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
        self.cov_diag_val_transition_model = cov_diag_val_transition_model
        self.transition_model_cov = np.diag(np.repeat(self.cov_diag_val_transition_model,self.obs_size))
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
    
    def mean_cov_so_t_gvnt_sag_t(self, sg_t, ag_t):
        # mean vector and covariance matrix to be used 
        # for Gaussian sampling of noisy state given ground truth state 
        mean_so_t_gauss = sg_t
        cov_so_t_gauss = np.diag(np.repeat(self.cov_diag_val_st_noise,len(mean_so_t_gauss)))
        return mean_so_t_gauss, cov_so_t_gauss

    def mean_cov_ao_t_gvnt_sag_t(self, sg_t, ag_t):
        # mean vector and covariance matrix to be used 
        # for Gaussian sampling of noisy state given ground truth state 
        mean_ao_t_gauss = ag_t
        cov_ao_t_gauss = np.diag(np.repeat(self.cov_diag_val_act_noise,len(mean_ao_t_gauss)))
        return mean_ao_t_gauss, cov_ao_t_gauss

    def mean_cov_sao_t_gvn_sag_t(self, sg_t, ag_t):
        mean_so_t_gauss, cov_so_t_gauss = self.mean_cov_so_t_gvnt_sag_t(sg_t, ag_t)
        mean_ao_t_gauss, cov_ao_t_gauss = self.mean_cov_ao_t_gvnt_sag_t(sg_t, ag_t)
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
    
    def intended_next_state(self, observation, action):
        # make input state current state. 
        full_state = np.append([self.init_qpos[0]],observation[:self.obs_size])
        qpos = np.array(full_state[:9])
        qvel = np.array(full_state[9:self.obs_size+1])
        self.set_state(qpos, qvel)
        # get next state
        result, _, _, _ = self.step(action)
        return result 

    def mean_cov_sg_t_gvn_sag_tmns1(self, sg_tmns1, ag_tmns1):
        ## mean cov for P(sg | sg_tmns1, ag_tmns1)  
        mean_sg_t_gauss = self.intended_next_state(sg_tmns1, ag_tmns1)
        cov_sg_t_gauss = self.transition_model_cov

        return mean_sg_t_gauss, cov_sg_t_gauss
    
    def min_non_zero_val(self, mat):
        # return smallest non zero element of matrix 
        sorted_flatnd_array = np.sort(mat.flatten())
        inds = np.nonzero(sorted_flatnd_array)
        if len(inds) > 0:
            return sorted_flatnd_array[inds[0][0]]
        else:
            return 0.0

    def hack_mats_covs_for_sum(self, cov1, cov2):
        # make smaller size matrix same as bigger size by filling zeroes for covariance
        # filling zeroes created singular matrix in subsequent product and inverse couldn't be computed
        # so filling min value
        if cov1.shape[0] > cov2.shape[0]: 
            tmp = np.identity(cov1.shape[0])*self.min_non_zero_val(cov2) 
            tmp[:cov2.shape[0],:cov2.shape[0]] = cov2.copy()
            cov2 = tmp
        else:
            tmp = np.identity(cov2.shape[0])*self.min_non_zero_val(cov1)  
            tmp[:cov1.shape[0],:cov1.shape[0]] = cov1.copy()
            cov1 = tmp

        return cov1, cov2

    def hack_mats_covs_for_product(self, cov1, cov2):
        # make smaller size matrix same as bigger size by filling min cov values for digaonal 
        if cov1.shape[0] > cov2.shape[0]: 
            tmp = np.identity(cov1.shape[0])*self.min_non_zero_val(cov2) 
            tmp[:cov2.shape[0],:cov2.shape[0]] = cov2.copy()
            cov2 = tmp
        else:
            tmp = np.identity(cov2.shape[0])*self.min_non_zero_val(cov1) 
            tmp[:cov1.shape[0],:cov1.shape[0]] = cov1.copy()
            cov1 = tmp

        return cov1, cov2

    def hack_vectors_means_for_product(self, mean1, mean2):
        # make smaller size vector same as bigger size by filling min mean value
        if len(mean1) > len(mean2): 
            tmp = np.ones(mean1.shape)*self.min_non_zero_val(mean2) 
            tmp[:len(mean2)] = mean2.copy()
            mean2 = tmp
        else:
            tmp = np.ones(mean2.shape)*self.min_non_zero_val(mean1)  
            tmp[:len(mean1)] = mean1.copy()
            mean1 = tmp

        return mean1, mean2

    def prod_gauss(self, mean1, cov1, mean2, cov2):
        # return mean and cov of product of Gaussians
        # ref: https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
        # section 8.1.8 in http://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf 

        cov1_sum, cov2_sum = self.hack_mats_covs_for_sum(cov1, cov2)
        inv_cov_12 = inv(cov1_sum + cov2_sum) 
        cov1_prd, cov2_prd = self.hack_mats_covs_for_product(cov1, cov2)
        covp = np.matmul(np.matmul(cov1_prd, inv_cov_12),cov2_prd) 
        # comparing with highest covariance value in config 
        # assert (covp < self.cov_diag_val_st_noise).all(), "covariance too high in product"
        mean1_prd, mean2_prd = self.hack_vectors_means_for_product(mean1, mean2)
        meanp = np.matmul(cov2_prd,np.matmul(inv_cov_12,mean1_prd))+np.matmul(cov1_prd,np.matmul(inv_cov_12,mean2_prd)) 
        
        return meanp, covp 

    def step_sa(self, s, a): # AdversarialTrainer > train_disc uses this method to create gound truth trajectory 
        return self.intended_next_state(s, a)

    def gibbs_sampling_mean_cov(self, list_inputs):
        '''
        Inputs:
        sg_tmns1, ag_tmns1 - prev timestep s-a pair of ground truth used in current iteration of discriminator update 
        mean_agt_gvn_sgt, cov_agt_gvn_sgt - mean and cov of Gaussian approximation imposed on P(. | sg_t) distribution output by generator 
        sg_t, ag_t - current timestep s-a pair of ground truth used in current iteration of discriminator update 

        return gibbs sampled sg_t, ag_t needed for logits computed in the next iteration of discriminator update 

        state should be sampled from Gaussian given by 
        P (s_g t| MB(s_g t)) =  P(sg | sg_tmns1, ag_tmns1) P (a_g t| s_g t) P (s_g tpls1| s_g t, a_g t) P (s_o t| s_g t, a_g t) P (a_o t| s_g t, a_g t)​ 
        P (a_g t| MB(a_g t)) =  P (a_g t| s_g t) P (s_g tpls1| s_g t, a_g t) P (s_o t| s_g t, a_g t) P (a_o t| s_g t, a_g t)​ 

        '''
        [sg_tmns1, ag_tmns1, mean_agt_gvn_sgt, cov_agt_gvn_sgt, sg_t, ag_t, sg_tpls1] = list_inputs
        st_time = time.time()
        if sg_tmns1 is not None:
            mean_sg_t_gvn_sag_tmns1, cov_sg_t_gvn_sag_tmns1 = self.mean_cov_sg_t_gvn_sag_tmns1(sg_tmns1, ag_tmns1)
        else:
            # GT_traj[j-1] is empty for j=0 timestep
            mean_sg_t_gvn_sag_tmns1, cov_sg_t_gvn_sag_tmns1 = None, None

        if sg_tpls1 is not None:
            mean_sg_tpls1_gvn_sag_t, cov_sg_tpls1_gvn_sag_t = self.mean_cov_sg_t_gvn_sag_tmns1(sg_t, ag_t)
            mean_so_t_gvn_sag_t, cov_so_t_gvn_sag_t, mean_ao_t_gvn_sag_t, cov_ao_t_gvn_sag_t = self.mean_cov_sao_t_gvn_sag_t(sg_t, ag_t)
        else:
            # GT_traj[j+1] is empty for len(GT_traj)-1 timestep
            mean_sg_tpls1_gvn_sag_t, cov_sg_tpls1_gvn_sag_t = None, None
            # a_g_t is -1 in for len(GT_traj)-1 timestep because expert_samples doesn't have an action
            mean_so_t_gvn_sag_t, cov_so_t_gvn_sag_t = self.mean_cov_so_t_gvnt_sag_t(sg_t, ag_t)
            mean_ao_t_gvn_sag_t, cov_ao_t_gvn_sag_t = None, None

        ## starting from right side of product of terms because last 4 terms are shared by two distribution
        ed_time1 = time.time() - st_time
        ed_time2 = 0.0
        if mean_ao_t_gvn_sag_t is not None:
            mean_sao_t_gvn_sag_t, cov_sao_t_gvn_sag_t = \
                self.prod_gauss(mean_so_t_gvn_sag_t, cov_so_t_gvn_sag_t, \
                                mean_ao_t_gvn_sag_t, cov_ao_t_gvn_sag_t)
            
            ed_time2 = time.time() - st_time - ed_time1
        else:
            # assume P (a_o t| s_g t, a_g t) = 1
            mean_sao_t_gvn_sag_t, cov_sao_t_gvn_sag_t = mean_so_t_gvn_sag_t, cov_so_t_gvn_sag_t

        ed_time3 = 0.0
        if mean_sg_tpls1_gvn_sag_t is not None:
            mean_next, cov_next = \
                self.prod_gauss(mean_sg_tpls1_gvn_sag_t, cov_sg_tpls1_gvn_sag_t, \
                                mean_sao_t_gvn_sag_t, cov_sao_t_gvn_sag_t) 

            ed_time3 = time.time() - st_time - ed_time1 - ed_time2
        else:
            # assume P (s_g tpls1| s_g t, a_g t) = 1
            mean_next, cov_next = mean_sao_t_gvn_sag_t, cov_sao_t_gvn_sag_t
        
        mean_Gs_a_g_t, cov_Gs_a_g_t = self.prod_gauss(mean_agt_gvn_sgt, cov_agt_gvn_sgt, mean_next, cov_next) 
        ed_time4 = time.time() - st_time - ed_time1 - ed_time2 - ed_time3

        # product of all 5 terms
        ed_time5 = 0.0
        if mean_sg_t_gvn_sag_tmns1 is not None:
            mean_Gs_s_g_t, cov_Gs_s_g_t = self.prod_gauss(mean_sg_t_gvn_sag_tmns1, cov_sg_t_gvn_sag_tmns1, mean_Gs_a_g_t, cov_Gs_a_g_t) 

            ed_time5 = time.time() - st_time - ed_time1 - ed_time2 - ed_time3 - ed_time4
        else:
            # assume P(sg | sg_tmns1, ag_tmns1) = 1
            mean_Gs_s_g_t, cov_Gs_s_g_t = mean_Gs_a_g_t, cov_Gs_a_g_t
            assert len(mean_Gs_a_g_t) == self.obs_size, "Wrong matrix size. Need modification in computing mean_Gs_a_g_t, cov_Gs_a_g_t "
            assert cov_Gs_a_g_t.shape[0] == self.obs_size, "Wrong matrix size. Need modification in computing mean_Gs_a_g_t, cov_Gs_a_g_t "
        
        ed_time6 = time.time() - st_time - ed_time1 - ed_time2 - ed_time3 - ed_time4 - ed_time5
        if len(mean_Gs_a_g_t) > self.act_size:
            # choose elements of self.act_size
            mean_Gs_a_g_t = mean_Gs_a_g_t[:self.act_size].copy()
            cov_Gs_a_g_t = cov_Gs_a_g_t[:self.act_size,:self.act_size].copy()

        if len(mean_Gs_s_g_t) > self.obs_size:
            # choose elements of self.obs_size
            mean_Gs_s_g_t = mean_Gs_s_g_t[:self.obs_size].copy()
            cov_Gs_s_g_t = cov_Gs_s_g_t[:self.obs_size,:self.obs_size].copy()

        # if mean is too far from current ground truth values, then adjust them
        # dev1 = np.linalg.norm(mean_Gs_s_g_t - sg_t)
        # dev2 = np.linalg.norm(mean_Gs_a_g_t - ag_t)
        # if dev1 > 0.01:
        #     mean_Gs_s_g_t = sg_t
        # if dev2 > 0.01:
        #     mean_Gs_a_g_t = ag_t 
        ed_time7 = time.time() - st_time - ed_time1 - ed_time2 - ed_time3 - ed_time4 - ed_time5 - ed_time6

        # print("ed_time1 {}, \ned_time2 {}, \ned_time3 {}, \ned_time4 {}, \ned_time5 {}, \ned_time6 {} \ned_time7 {}".format(\
        #     ed_time1, ed_time2, ed_time3, ed_time4, ed_time5, ed_time6, ed_time7)) 

        return mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t
