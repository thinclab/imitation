"""Core code for adversarial imitation learning, shared between GAIL and AIRL."""
import abc
import collections
import dataclasses
import logging
import os
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type, Iterable, Union

import numpy as np
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util


def compute_train_stats(
    disc_logits_expert_is_high: th.Tensor,
    labels_expert_is_one: th.Tensor,
    disc_loss: th.Tensor,
) -> Mapping[str, float]:
    """Train statistics for GAIL/AIRL discriminator.

    Args:
        disc_logits_expert_is_high: discriminator logits produced by
            `AdversarialTrainer.logits_expert_is_high`.
        labels_expert_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        A mapping from statistic names to float values.
    """
    with th.no_grad():
        # Logits of the discriminator output; >0 for expert samples, <0 for generator.
        bin_is_generated_pred = disc_logits_expert_is_high < 0
        # Binary label, so 1 is for expert, 0 is for generator.
        bin_is_generated_true = labels_expert_is_one == 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = float(th.sum(int_is_generated_true))
        n_labels = float(len(labels_expert_is_one))
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - th.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            expert_acc = _n_pred_expert / float(n_expert)

        _n_pred_gen = th.sum(th.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = th.distributions.Bernoulli(logits=disc_logits_expert_is_high)
        entropy = th.mean(label_dist.entropy())

    pairs = [
        ("disc_loss", float(th.mean(disc_loss))),
        # accuracy, as well as accuracy on *just* expert examples and *just*
        # generated examples
        ("disc_acc", float(acc)),
        ("disc_acc_expert", float(expert_acc)),
        ("disc_acc_gen", float(generated_acc)),
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        ("disc_entropy", float(entropy)),
        # true number of expert demos and predicted number of expert demos
        ("disc_proportion_expert_true", float(pct_expert)),
        ("disc_proportion_expert_pred", float(pct_expert_pred)),
        ("n_expert", float(n_expert)),
        ("n_generated", float(n_generated)),
    ]  # type: Sequence[Tuple[str, float]]
    return collections.OrderedDict(pairs)


class AdversarialTrainer(base.DemonstrationAlgorithm[types.Transitions]):
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        n_disc_updates_per_round: int = 2,
        log_dir: str = "output/",
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        gen_train_timesteps: Optional[int] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        allow_variable_horizon: bool = False,
        threshold_stop_Gibbs_sampling: float,
        num_iters_Gibbs_sampling: int
    ):
        """Builds AdversarialTrainer.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensors as input and computes a reward signal.
            n_disc_updates_per_round: The number of discriminator updates after each
                round of generator updates in AdversarialTrainer.learn().
            log_dir: Directory to store TensorBoard logs, plots, etc. in.
            disc_opt_cls: The optimizer for discriminator training.
            disc_opt_kwargs: Parameters for discriminator training.
            gen_train_timesteps: The number of steps to train the generator policy for
                each iteration. If None, then defaults to the batch size (for on-policy)
                or number of environments (for off-policy).
            gen_replay_buffer_capacity: The capacity of the
                generator replay buffer (the number of obs-action-obs samples from
                the generator that can be stored). By default this is equal to
                `gen_train_timesteps`, meaning that we sample only from the most
                recent batch of generator samples.
            custom_logger: Where to log to; if None (default), creates a new logger.
            init_tensorboard: If True, makes various discriminator
                TensorBoard summaries.
            init_tensorboard_graph: If both this and `init_tensorboard` are True,
                then write a Tensorboard graph summary to disk.
            debug_use_ground_truth: If True, use the ground truth reward for
                `self.train_env`.
                This disables the reward wrapping that would normally replace
                the environment reward with the learned reward. This is useful for
                sanity checking that the policy training is functional.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
        """
        self.demo_batch_size = demo_batch_size
        self._demo_data_loader = None
        self._endless_expert_iterator = None

        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo
        self._reward_net = reward_net.to(gen_algo.device)
        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = self._disc_opt_cls(
            self._reward_net.parameters(),
            **self._disc_opt_kwargs,
        )

        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        venv = self.venv_buffering = wrappers.BufferingWrapper(self.venv)

        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_wrapped = venv
            self.gen_callback = None
        else:
            venv = self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
                venv,
                reward_fn=self.reward_train.predict_processed,
            )
            self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        if gen_train_timesteps is None:
            gen_algo_env = self.gen_algo.get_env()
            assert gen_algo_env is not None
            self.gen_train_timesteps = gen_algo_env.num_envs
            if hasattr(self.gen_algo, "n_steps"):  # on policy
                self.gen_train_timesteps *= self.gen_algo.n_steps
        else:
            self.gen_train_timesteps = gen_train_timesteps

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_train_timesteps
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity,
            self.venv,
        )

        self.threshold_stop_Gibbs_sampling = threshold_stop_Gibbs_sampling
        self.num_iters_Gibbs_sampling = num_iters_Gibbs_sampling

    @property
    def policy(self) -> policies.BasePolicy:
        return self.gen_algo.policy

    @abc.abstractmethod
    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting expert, and a low value corresponds to
        predicting generator.

        Args:
            state: state at time t, of shape `(batch_size,) + state_shape`.
            action: action taken at time t, of shape `(batch_size,) + action_shape`.
            next_state: state at time t+1, of shape `(batch_size,) + state_shape`.
            done: binary episode completion flag after action at time t,
                of shape `(batch_size,)`.
            log_policy_act_prob: log probability of generator policy taking
                `action` at time t.

        Returns:
            Discriminator logits of shape `(batch_size,)`. A high output indicates an
            expert-like transition.
        """  # noqa: DAR202

    @property
    @abc.abstractmethod
    def reward_train(self) -> reward_nets.RewardNet:
        """Reward used to train generator policy."""

    @property
    @abc.abstractmethod
    def reward_test(self) -> reward_nets.RewardNet:
        """Reward used to train policy at "test" time after adversarial training."""

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
            dict(shuffle=False, drop_last=True)
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Optional[Mapping[str, float]]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            if expert_samples is None:
                expert_samples = self._next_expert_batch()

            if gen_samples is None:
                if self._gen_replay_buffer.size() == 0:
                    raise RuntimeError(
                        "No generator samples for training. " "Call `train_gen()` first.",
                    )
                gen_samples = self._gen_replay_buffer.sample(self.demo_batch_size)
                gen_samples = types.dataclass_quick_asdict(gen_samples)

            if not self.threshold_stop_Gibbs_sampling:
                # code without Gibbs sampling 

                # compute loss
                batch = self._make_disc_train_batch(
                    gen_samples=gen_samples,
                    expert_samples=expert_samples,
                )
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
            else:
                # disc_logits to be computed via Gibbs sampling 

                import time 
                from torch.distributions import Categorical 
                import git 
                import math 
                import gym
                import itertools

                start_tm_disc_train = time.time()

                repo = git.Repo('.', search_parent_directories=True) 
                git_home = repo.working_tree_dir 

                filename=str(git_home)+"/for_debugging/troubleshooting_gibbs_sampling.txt" 
                writer = open(filename, "a") 
                

                def get_state_space_upperlimit():
                    '''
                    if venv state space is discrete, then return observation space limit
                    if it's continuous, then return total number of partitions of state space 
                    '''
                    if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                        return self.venv.observation_space.n
                    elif isinstance(self.venv.observation_space, gym.spaces.Box): 
                        num_parts = len(self.venv.env_method(method_name='state_space_partitions',indices=0)[0])
                        return num_parts
                    else:
                        raise TypeError("Unsupported type of state space")

                def create_gibbs_sampler(expert_samples_batch,GT_traj,combinations_sa_tuples,writer):
                    '''
                    GT_traj has length expert_samples_batch['obs']+1 
                    for index j in expert_samples_batch['obs'], create corresponding list of probability values.  
                    This creationg is possible all timesteps of expert_samples_batch['obs'] because we have 
                    expert_samples_batch['acts'] value for each time step. But this list can't be created for
                    expert_samples_batch['nextobs'] last timestep (same as last timestep of GT_traj) because there 
                    is no last action in input demonstration (only a last state). For such edge cases, we 
                    assume prob values 1 for observation model part. 
                    
                    sa_distr_samples['obs_sa_distr'][j] and sa_distr_samples['nxtobs_sa_distr'][j] separate out that 
                    last timestep 

                    '''

                    # uncomment for discretized mountain car 
                    # def policy_a_giv_s_cont_st_disc_act(act,partition_sg):
                    #     # integration of P(act|s_g,t) over partition_sg, term B in Gibbs sampler draft on overleaf
                        
                    #     samples_sg,size_Sg = self.venv.env_method(method_name='discrete_samples_to_estimate_integral',indices=0,ind=partition_sg)[0]
                        
                    #     a_th = th.as_tensor([act]*len(samples_sg), device=self.gen_algo.device) 
                        
                    #     sg_cont_th = th.as_tensor(np.array(samples_sg), device=self.gen_algo.device) 
                    #     sum_monte_carlo = self.policy.prob_acts(obs=sg_cont_th,actions=a_th).sum().item() 
                    
                    #     return (size_Sg * sum_monte_carlo * 1/(len(samples_sg)))

                    start_tm = time.time()

                    if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                        # temp storage of traj specific list of probs_sa_gt_sa_j 
                        sa_distr_trajs = np.zeros((len(GT_traj),len(combinations_sa_tuples)),dtype=float) 

                        for j in range(len(GT_traj)):
                            start_tm2 = time.time()
                            
                            # list of probabilities specific to time step j 
                            probs_sa_gt_sa_j = [0.0]*len(combinations_sa_tuples) 

                            for s in range(get_state_space_upperlimit()):
                                for a in range(self.venv.action_space.n):
                                    
                                    skip_next_steps = False 
                                    start_tm3 = time.time()
                                    if j==len(GT_traj)-1:
                                        # GT_traj[j+1] is empty for len(GT_traj)-1 index
                                        P_nexts_s_a = 1.0 
                                    else: 
                                        if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                                            P_nexts_s_a = self.venv.env_method(method_name='P_sasp',indices=0,s=s,a=a,sp=GT_traj[j+1][0])[0] 
                                        # elif isinstance(self.venv.observation_space, gym.spaces.Box): 
                                        #     # for continuous state discrete action domain, we created separate method. 
                                        #     # this call has s as a partition for state space but sp as a continuous state
                                        #     P_nexts_s_a = self.venv.env_method(method_name='P_sasp2_no_intgrl',indices=0,s=s,a=a,sp=GT_traj[j+1][0])[0] 
                                        else:
                                            raise TypeError("Unsupported type of state space")
                                    tm_P_nexts_s_a = (time.time()-start_tm3)/60 
                                    if P_nexts_s_a==0:
                                        skip_next_steps = True
                                        probs_sa_gt_sa_j[combinations_sa_tuples.index((s,a))] = 0

                                    if not skip_next_steps:
                                        if j==0:
                                            P_s_prevs_preva = 1.0
                                        else: 
                                            # for continuous state space, this call has s as a continuous state but sp as a partition for state space
                                            P_s_prevs_preva = self.venv.env_method(method_name='P_sasp',indices=0,s=GT_traj[j-1][0],a=GT_traj[j-1][1],sp=s)[0]
                                        tm_P_s_prevs_preva = (time.time()-start_tm3)/60 - tm_P_nexts_s_a
                                        if P_s_prevs_preva==0:
                                            skip_next_steps = True
                                            probs_sa_gt_sa_j[combinations_sa_tuples.index((s,a))] = 0

                                    if not skip_next_steps:
                                        if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                                            # if state space is discrete
                                            s_th = th.as_tensor([s], device=self.gen_algo.device) 
                                            a_th = th.as_tensor([a], device=self.gen_algo.device) 
                                            policy_a_giv_s = self.policy.prob_acts(obs=s_th,actions=a_th)[0].item() 
                                        # elif isinstance(self.venv.observation_space, gym.spaces.Box): 
                                        #     # if state space is continous
                                        #     policy_a_giv_s = policy_a_giv_s_cont_st_disc_act(a,s)
                                        else:
                                            raise TypeError("Unsupported type of state space")
                                        tm_policy_a_giv_s = (time.time()-start_tm3)/60 - tm_P_s_prevs_preva - tm_P_nexts_s_a
                                        if policy_a_giv_s==0:
                                            skip_next_steps = True
                                            probs_sa_gt_sa_j[combinations_sa_tuples.index((s,a))] = 0

                                    if not skip_next_steps:
                                        if j==len(GT_traj)-1: 
                                            # obsvd_traj.acts is empty at len(GT_traj)-1
                                            P_obssa_GTsa = 1.0
                                        else:
                                            # for continuous state space, this call has sg as a partition for state space but so as a continuous state 
                                            P_obssa_GTsa = self.venv.env_method(method_name='obs_model',indices=0,sg=s,ag=a,so=expert_samples_batch['obs'][j].numpy(),ao=expert_samples_batch['acts'][j].item())[0] 
                                        tm_P_obssa_GTsa = (time.time()-start_tm3)/60 - tm_P_s_prevs_preva - tm_P_nexts_s_a - tm_policy_a_giv_s

                                        probs_sa_gt_sa_j[combinations_sa_tuples.index((s,a))] = P_s_prevs_preva * \
                                            policy_a_giv_s * P_nexts_s_a * P_obssa_GTsa 
                                    
                                        # print("\ntm_P_s_prevs_preva = {}\ntm_P_nexts_s_a = {}\ntm_P_obssa_GTsa = {}\ntm_policy_a_giv_s = {}".format(tm_P_s_prevs_preva,tm_P_nexts_s_a,tm_P_obssa_GTsa,tm_policy_a_giv_s))
                                        # print("estimated time per call to gibbs_sampler = {} hours".format((time.time()-start_tm3)*get_state_space_upperlimit()*self.venv.action_space.n*len(GT_traj)*1/3600))
                            
                            # Gibbs sampling distribution for time step j of ground truth trajectory 
                            sa_distr_trajs[j]=np.array(probs_sa_gt_sa_j)

                    elif isinstance(self.venv.observation_space, gym.spaces.Box):
                            
                        def mean_cov_agt_gvn_sgt(s_g_t):
                            # sample a_g_t 10000 times and use those samples to make mean and cov of 
                            # Gaussian approximation of generator's P(. | s_g_t) distribution
                            sg_cont_th = th.as_tensor(np.repeat([s_g_t], 10000, axis=0), device=self.gen_algo.device) 
                            samples_gauss = self.policy.predict(sg_cont_th,deterministic=False)[0]
                            mean_agt_gvn_sgt = np.mean(samples_gauss, axis=0)
                            cov_agt_gvn_sgt = np.cov(np.transpose(samples_gauss))
                            return mean_agt_gvn_sgt,cov_agt_gvn_sgt
                            
                        # temp storage of traj specific list of probs_sa_gt_sa_j 
                        # for each timestep t, we store mean and cov of P(s_g_t | MB(s_g_t)) and P(a_g_t|MB(a_g_t))
                        sa_distr_trajs = []

                        for j in range(len(GT_traj)):
                            st_time2 = time.time() 
                            sg_tmns1, ag_tmns1, sg_t, ag_t, sg_tpls1 = \
                                None, None, GT_traj[j][0], GT_traj[j][1] , None 
                            mean_agt_gvn_sgt, cov_agt_gvn_sgt = mean_cov_agt_gvn_sgt(sg_t)

                            ed_time1 = time.time() - st_time2
                            if j!=len(GT_traj)-1:
                                sg_tpls1 = GT_traj[j+1][0]

                            if j!=0:
                                sg_tmns1, ag_tmns1 = GT_traj[j-1][0], GT_traj[j-1][1]

                            mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t = \
                                self.venv.env_method(method_name='gibbs_sampling_mean_cov',indices=[0]*self.venv.num_envs,\
                                sg_tmns1=sg_tmns1, ag_tmns1=ag_tmns1, mean_agt_gvn_sgt=mean_agt_gvn_sgt, cov_agt_gvn_sgt=\
                                cov_agt_gvn_sgt,sg_t=sg_t, ag_t=ag_t, sg_tpls1=sg_tpls1)[0]
                            ed_time2 = time.time() - st_time2 - ed_time1
                            sa_distr_trajs.append([mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t])
                            print("create_gibbs_sampler ed_time1 {}, ed_time2 {}".format(ed_time1,ed_time2))

                    else:
                        raise TypeError("Unsupported type of state space")

                    # print("time taken to create sa_distr_trajs: {} minutes ".format((time.time()-start_tm)/60)) 
                    writer.write("time taken to create sa_distr_trajs: {} minutes \n".format((time.time()-start_tm)/60)) 

                    sa_distr_samples = {}
                    sa_distr_samples["obs_sa_distr"] = sa_distr_trajs[:-1]
                    sa_distr_samples["nxtobs_sa_distr"] = sa_distr_trajs[1:]

                    return sa_distr_samples  

                def sample_sa_gibbs_sampler(probs_sa_gt_sa_j,combinations_sa_tuples,writer):
                    '''
                    internal method to sample from a given list of probs
                    for discrete states, returned value s in state index
                    for continuous states, returned value s is index of partition of state space 
                    '''

                    s, a = None, None
                    total_mass = 0
                    for pr in probs_sa_gt_sa_j:
                        total_mass += pr
                    
                    # It is possible and valid that dict_gt_sa has all values 0? Most of cases are coming 0. 
                    # Skip the case where all values in probs_sa_gt_sa_j are zero
                    if (total_mass != 0):
                        # wr_str = "sample_sa: probs_sa_gt_sa_j all values aren't 0s"
                        # writer.write(str(wr_str)+"\n")
                        
                        if len(set(probs_sa_gt_sa_j)) == 1: 
                            # sample and replace GT s-a in GT_traj
                            norm = [float(i)/sum(probs_sa_gt_sa_j) for i in probs_sa_gt_sa_j]
                            '''
                            Distr Example::

                                >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
                                >>> m.sample()  # equal probability of 0, 1, 2, 3
                                tensor(3)
                            '''
                            m = Categorical(th.tensor(norm))
                            sampled_ind = m.sample().item()
                            s, a = combinations_sa_tuples[sampled_ind]
                            # print("sample_sa: s,a in perceived traj changed by uniform sampling")
                            # wr_str = "sample_sa: s,a in perceived traj changed by uniform sampling"
                            # writer.write(str(wr_str)+"\n")
                        else:
                            max_value = max(probs_sa_gt_sa_j)
                            max_index = probs_sa_gt_sa_j.index(max_value)
                            s, a = combinations_sa_tuples[max_index]
                            # print("sample_sa: s,a in perceived traj changed by pick max prob index")
                            # wr_str = "sample_sa: s,a in perceived traj changed by pick max prob index"
                            # writer.write(str(wr_str)+"\n")
                    else:
                        # wr_str = "sample_sa: probs_sa_gt_sa_j all values are 0s"
                        # writer.write(str(wr_str)+"\n")
                        pass

                    return s, a

                def get_state_from_sampled_index(s):
                    return s 
                    # uncomment for using mountain car
                    # if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                    #     return s 
                    # elif isinstance(self.venv.observation_space, gym.spaces.Box): 
                    #     cont_st = self.venv.env_method(method_name='sample_random_state_from_partition',indices=0,ind=s)[0] 
                    #     return cont_st
                    # else:
                    #     raise TypeError("Unsupported type of state space")

                combinations_sa_tuples = None 
                if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                    list_s = list(range(get_state_space_upperlimit()))
                    list_a = list(range(self.venv.action_space.n))
                    start_tm = time.time()
                    combinations_sa_tuples = list(itertools.product(list_s,list_a))
                    print("time taken to create combinations_sa_tuples: {} minutes ".format((time.time()-start_tm)/60))

                # create GT_trajs 
                # simulate random ground truth trajectories from learned policy 
                GT_traj = np.empty((len(expert_samples['obs'])+1,2),dtype=int)
                init_obs = self.venv.reset()
                GT_traj = np.array([[init_obs[0],None]])
                for j in range(len(expert_samples['obs'])):
                    a, _ = self.policy.predict(GT_traj[j][0],deterministic=False)
                    if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                        GT_traj[j][1] = a.item(0)
                    else:
                        GT_traj[j][1] = a
                    ret_tuples = self.venv.env_method(method_name='step_sa',indices=[0]*self.venv.num_envs,s=GT_traj[j][0],a=GT_traj[j][1])
                    ns = ret_tuples[0] 
                    GT_traj = np.vstack((GT_traj,np.array([ns, -1]))) 

                # sa_distr_samples = self._next_sa_distr_batch() 
                normed_delta_avg_disc_logits = math.inf
                import torch as th
                sum_avg_disc_logits = th.tensor( [0.0]*(len(expert_samples['obs'])+len(gen_samples['obs'])), device=self.gen_algo.device)
                ind_avg = 0
                avg_disc_logits = th.tensor( [0.0]*(len(expert_samples['obs'])+len(gen_samples['obs'])) )
                last_avg_disc_logits = th.tensor( [0.0]*(len(expert_samples['obs'])+len(gen_samples['obs'])), device=self.gen_algo.device)
                # to compute logits, we only need sampled ground truth in same format as expert_samples 
                gibbs_sampled_expert_samples = {}
                for key,val in expert_samples.items():
                    if key!='infos':
                        gibbs_sampled_expert_samples[key] = val.detach().clone()
                    else: 
                        gibbs_sampled_expert_samples[key] = val.copy()

                '''
                use a loop for following with a stopping criteria based on delta in average disc_logits array values
                    create gibbs sampler distribution for each timestep in expert_samples
                    for each step in expert_samples, use obs_sa_distr for current time step to create GT gibbs sampled expert samples. 
                    (Do not modify observations in expert_samples)
                    use Gibbs sampled expert_samples to compute disc_logits
                    add disc_logits array to sum so far
                    compute running average of logits
                    compute delta change in average w.r.t previous iteration 

                '''
                if isinstance(self.venv.observation_space, gym.spaces.Box): 
                    obs_size = self.venv.get_attr('obs_size')[0]
                    act_size = self.venv.get_attr('act_size')[0]

                while normed_delta_avg_disc_logits > self.threshold_stop_Gibbs_sampling and \
                    ind_avg <= self.num_iters_Gibbs_sampling:
    
                    sa_distr_samples = create_gibbs_sampler(expert_samples,GT_traj,combinations_sa_tuples,writer) 

                    #################################################################################
                    # gibbs sample 'potential ground truth' and get it formatted for computing logits 
                    # sampled 'potential ground truth' should not replace original noisy observations 
                    for j in range(len(expert_samples['obs'])):
                        
                        # sampling must happen specifically for index j
                        if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                            probs_sa_gt_sa_j = sa_distr_samples['obs_sa_distr'][j].tolist() 
                            s,a = sample_sa_gibbs_sampler(probs_sa_gt_sa_j,combinations_sa_tuples,writer)
                        elif isinstance(self.venv.observation_space, gym.spaces.Box):
                            [mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t] = sa_distr_samples['obs_sa_distr'][j]
                            s = np.random.multivariate_normal(mean_Gs_s_g_t, cov_Gs_s_g_t, (1))[0]
                            assert len(s)==obs_size, "Sampled ground truth state size {} not eq obs_size {}".format(len(s),obs_size)
                            a = np.random.multivariate_normal(mean_Gs_a_g_t, cov_Gs_a_g_t, (1))[0]
                            assert len(a)==act_size, "Sampled ground truth action size {} not eq act_size {}".format(len(a),act_size)
                        else:
                            raise TypeError("Unsupported type of state space")

                        # to compute logits, we only need sampled ground truth in same format as expert_samples 
                        if s is not None:
                            st = get_state_from_sampled_index(s)

                            # if sampling valid, get samples ready for computing logits 
                            gibbs_sampled_expert_samples['obs'][j] = th.tensor(st) if isinstance(st, int) else th.from_numpy(st)
                            gibbs_sampled_expert_samples['acts'][j] = th.tensor(a)

                            # update next obs of prev step based on obs sampled for current step 
                            if j>0:
                                gibbs_sampled_expert_samples['next_obs'][j-1] = gibbs_sampled_expert_samples['obs'][j]

                            # update ground truth trajectory GT_traj to get Gibbs sampler for next iteration of while loop 
                            GT_traj[j][0] = st
                            GT_traj[j][1] = a

                    # sampling last state in 'next_obs' part to gibbs_sampled_expert_samples 
                    if isinstance(self.venv.observation_space, gym.spaces.Discrete):
                        probs_sa_gt_sa_j = sa_distr_samples['nxtobs_sa_distr'][-1].tolist() 
                        last_s,last_a = sample_sa_gibbs_sampler(probs_sa_gt_sa_j,combinations_sa_tuples,writer) 
                    elif isinstance(self.venv.observation_space, gym.spaces.Box):
                        [mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t] = sa_distr_samples['nxtobs_sa_distr'][-1]
                        last_s = np.random.multivariate_normal(mean_Gs_s_g_t, cov_Gs_s_g_t, (1))[0]
                        assert len(last_s)==obs_size, "Sampled ground truth state last_s size {} not eq obs_size {}".format(len(s),obs_size)
                    else:
                        raise TypeError("Unsupported type of state space")

                    if last_s is not None: 
                        st = get_state_from_sampled_index(last_s) 
                        gibbs_sampled_expert_samples['next_obs'][-1] = th.tensor(st) if isinstance(st, int) else th.from_numpy(st)

                        # update ground truth trajectory GT_traj to get Gibbs sampler for next iteration of while loop 
                        GT_traj[len(expert_samples['obs'])][0] = st

                    #################################################################################

                    batch = self._make_disc_train_batch(
                        gen_samples=gen_samples, expert_samples=gibbs_sampled_expert_samples
                    )
                    disc_logits = self.logits_expert_is_high(
                        batch["state"],
                        batch["action"],
                        batch["next_state"],
                        batch["done"],
                        batch["log_policy_act_prob"],
                    )

                    # get running average 
                    ind_avg += 1
                    sum_avg_disc_logits += disc_logits
                    avg_disc_logits = sum_avg_disc_logits/ind_avg

                    # norm of change in average w.r.t last iteration 
                    delta_avg_disc_logits = (avg_disc_logits - last_avg_disc_logits)
                    normed_delta_avg_disc_logits = th.linalg.norm(delta_avg_disc_logits, ord=1).cpu().data.numpy() 

                    if True: #ind_avg == 1 or ind_avg%100 == 0:
                        # print("iteration {} normed_delta_avg_disc_logits {}".format(ind_avg, normed_delta_avg_disc_logits))
                        wr_str = "iteration {} normed_delta_avg_disc_logits {}".format(ind_avg, normed_delta_avg_disc_logits) 
                        writer.write(str(wr_str)+"\n") 

                    # update lat_avg variable to get normed delta for next iteration 
                    last_avg_disc_logits = avg_disc_logits

                wr_str = " normed_delta_avg_disc_logits converged in {}".format(round(( time.time()- start_tm_disc_train)/60,2)) 
                writer.write(str(wr_str)+"\n")    
                writer.close()
                
                print("Gibbs sampled averaged discimantor logits ",avg_disc_logits)
                print(" normed_delta_avg_disc_logits converged in {}".format(round(( time.time()- start_tm_disc_train)/60,2)))
                disc_logits = avg_disc_logits
            
            loss = F.binary_cross_entropy_with_logits(
                disc_logits,
                batch["labels_expert_is_one"].float(),
            )

            # do gradient step
            self._disc_opt.zero_grad()
            loss.backward()
            self._disc_opt.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            import torch as th
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}

        with self.logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                **learn_kwargs,
            )
            self._global_step += 1

        gen_trajs, ep_lens = self.venv_buffering.pop_trajectories()
        self._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every "round" consists of a call to `train_gen(self.gen_train_timesteps)`,
        a call to `train_disc`, and finally a call to `callback(round)`.

        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Args:
            total_timesteps: An upper bound on the number of transitions to sample
                from the environment during training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_train_timesteps)
            for _ in range(self.n_disc_updates_per_round):
                with networks.training(self.reward_train):
                    # switch to training mode (affects dropout, normalization)
                    self.train_disc()
            if callback:
                callback(r)
            self.logger.dump(self._global_step)

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.reward_train.device)

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
    ) -> Optional[th.Tensor]:
        """Evaluates the given actions on the given observations.

        Args:
            obs_th: A batch of observations.
            acts_th: A batch of actions.

        Returns:
            A batch of log policy action probabilities.
        """
        if isinstance(self.policy, policies.ActorCriticPolicy):
            # policies.ActorCriticPolicy has a concrete implementation of
            # evaluate_actions to generate log_policy_act_prob given obs and actions.
            _, log_policy_act_prob_th, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )
        elif isinstance(self.policy, sac_policies.SACPolicy):
            gen_algo_actor = self.policy.actor
            assert gen_algo_actor is not None
            # generate log_policy_act_prob from SAC actor.
            mean_actions, log_std, _ = gen_algo_actor.get_action_dist_params(obs_th)
            distribution = gen_algo_actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
            )
            # SAC applies a squashing function to bound the actions to a finite range
            # `acts_th` need to be scaled accordingly before computing log prob.
            # Scale actions only if the policy squashes outputs.
            assert self.policy.squash_output
            scaled_acts_th = self.policy.scale_action(acts_th)
            log_policy_act_prob_th = distribution.log_prob(scaled_acts_th)
        else:
            return None
        return log_policy_act_prob_th

    def _make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Mapping[str, th.Tensor]:
        """Build and return training batch for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Returns:
            The training batch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        # if expert_samples is None:
        #     expert_samples = self._next_expert_batch()

        # if gen_samples is None:
        #     if self._gen_replay_buffer.size() == 0:
        #         raise RuntimeError(
        #             "No generator samples for training. " "Call `train_gen()` first.",
        #         )
        #     gen_samples = self._gen_replay_buffer.sample(self.demo_batch_size)
        #     gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.demo_batch_size):
            raise ValueError(
                "Need to have exactly self.demo_batch_size number of expert and "
                "generator samples, each. "
                f"(n_gen={n_gen} n_expert={n_expert} "
                f"demo_batch_size={self.demo_batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        n_samples = n_expert + n_gen
        assert n_expert == len(expert_samples["acts"])
        assert n_expert == len(expert_samples["next_obs"])
        assert n_gen == len(gen_samples["acts"])
        assert n_gen == len(gen_samples["next_obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        # notice that the labels use the convention that expert samples are
        # labelled with 1 and generator samples with 0.
        labels_expert_is_one = np.concatenate(
            [np.ones(n_expert, dtype=int), np.zeros(n_gen, dtype=int)],
        )

        # Calculate generator-policy log probabilities.
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            acts_th = th.as_tensor(acts, device=self.gen_algo.device)
            log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
            if log_policy_act_prob is not None:
                assert len(log_policy_act_prob) == n_samples
                log_policy_act_prob = log_policy_act_prob.reshape((n_samples,))
            del obs_th, acts_th  # unneeded

        obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
            obs,
            acts,
            next_obs,
            dones,
        )
        batch_dict = {
            "state": obs_th,
            "action": acts_th,
            "next_state": next_obs_th,
            "done": dones_th,
            "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
            "log_policy_act_prob": log_policy_act_prob,
        }

        return batch_dict
