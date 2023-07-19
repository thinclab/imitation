"""Configuration for imitation.scripts.train_adversarial."""

import sacred

from imitation.rewards import reward_nets
from imitation.scripts.common import common, demonstrations, reward, rl, train
import stable_baselines3 as sb3


train_adversarial_ex = sacred.Experiment(
    "train_adversarial",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        train.train_ingredient,
    ],
)


@train_adversarial_ex.config
def defaults():
    show_config = False

    total_timesteps = int(1e6)  # Num of environment transitions to sample
    # total_timesteps = int(5e3) 
    algorithm_kwargs = dict(
        demo_batch_size=1024,  # Number of expert samples per discriminator update
        n_disc_updates_per_round=4,  # Num discriminator updates per generator round
    )
    algorithm_specific = {}  # algorithm_specific[algorithm] is merged with config

    checkpoint_interval = 0  # Num epochs between checkpoints (<0 disables)
    agent_path = None  # Path to load agent from, optional.
    demonstration_policy_path = None
    wdGibbsSamp = False
    threshold_stop_Gibbs_sampling = 0.05
    num_iters_Gibbs_sampling = 50
    sa_distr_read = False
    eval_n_timesteps = 2000
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    n_episodes_eval = -1 # used in train.eval_policy  
    env_make_kwargs = {}
    noise_insertion_raw_data = False
    rl_batch_size = None

@train_adversarial_ex.config
def aliases_default_gen_batch_size(algorithm_kwargs, rl):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.
    algorithm_kwargs["gen_replay_buffer_capacity"] = rl["batch_size"]


# Shared settings

MUJOCO_SHARED_LOCALS = dict(rl=dict(rl_kwargs=dict(ent_coef=0.1)))

ANT_SHARED_LOCALS = dict(
    total_timesteps=int(3e7),
    algorithm_kwargs=dict(shared=dict(demo_batch_size=8192)),
    rl=dict(batch_size=16384),
)


# Classic RL Gym environment named configs


@train_adversarial_ex.named_config
def acrobot():
    env_name = "Acrobot-v1"
    algorithm_kwargs = {"allow_variable_horizon": True}
    n_episodes_eval = -1


@train_adversarial_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    total_timesteps = int(1.4e6)


@train_adversarial_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@train_adversarial_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")

# Custom continuous state environment 

@train_adversarial_ex.named_config
def soContSpaces():
    common = dict(env_name="soContSpaces-v1", num_vec=32)
    # common = dict(env_name="soContSpaces-v1",num_vec=1)
    # rl_batch_size = 1024 # try setting it from call to train adversarial 
    # total_timesteps = int(1e5)
    # without specifying details of reward network, the reward.py config_hook should pick reward_nets.BasicShapedRewardNet (32, 32 default size)
    # algorithm_kwargs = dict(
    #     # Number of discriminator updates after each round of generator updates n_disc_updates_per_round
    #     n_disc_updates_per_round=4,
    #     # Equivalent to no replay buffer if batch size is the same gen_replay_buffer_capacity
    #     # gen_replay_buffer_capacity=16384,
    #     demo_batch_size=128, # <= size of rollout
    # )
    # eval_n_timesteps = algorithm_kwargs['demo_batch_size'] # minimum demo size we want for i2rl sessions
    # max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    n_episodes_eval = -1 # used in train.eval_policy  
    # args for initializing gym env class 
    # env_make_kwargs = {
    #     'rollout_path': demonstrations['rollout_path'], 
    #     'full_observable': True, 
    #     'max_steps': 400
    #     }


# Standard MuJoCo Gym environment named configs

@train_adversarial_ex.named_config
def seals_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    common = dict(env_name="seals/Ant-v0")


@train_adversarial_ex.named_config
def half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="HalfCheetah-v2")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    algorithm_specific = dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates n_disc_updates_per_round
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same gen_replay_buffer_capacity
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    )


@train_adversarial_ex.named_config
def half_cheetah_mdfd_weights():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="imitationNM/HalfCheetahEnvMdfdWeights-v0",num_vec = 2)
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))
    # algorithm_specific = dict(
    #     airl=dict(total_timesteps=int(5e6)),
    #     gail=dict(total_timesteps=int(8e6)),
    # )
    reward = dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        # demo_batch_size=8192,
        demo_batch_size = 128, # trying lower size to reduce session time with Gibbs sampling
    )
    eval_n_timesteps = algorithm_kwargs['demo_batch_size'] # minimum demo size we want for i2rl sessions
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    n_episodes_eval = -1 # used in train.eval_policy  
    env_make_kwargs = {
        'cov_diag_val_transition_model': 0.0001, 
        'cov_diag_val_st_noise': 0.1,
        'cov_diag_val_act_noise': 0.00001, 
        'noise_insertion': False}

@train_adversarial_ex.named_config
def hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name = "imitationNM/Hopper-v3", num_vec = 8) 
    rl = dict(
            rl_cls=sb3.PPO,
            batch_size=512*1, # desired_n_steps*num_vec from common.py half_cheetah_mdfd_weights
            rl_kwargs=dict(
                batch_size=32,
                gamma=0.999,
                learning_rate=9.80828e-05,
                ent_coef=0.00229519,
                clip_range=0.2,
                n_epochs=5,
                gae_lambda=0.99,
                max_grad_norm=0.7,
                vf_coef=0.835671)
            )
    algorithm_kwargs = dict(
        demo_batch_size = 32, 
        allow_variable_horizon = True,
    )
    eval_n_timesteps = algorithm_kwargs['demo_batch_size'] # minimum demo size we want for i2rl sessions
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    n_episodes_eval = -1 # used in train.eval_policy  
    env_make_kwargs = {
        'cov_diag_val_transition_model': 0.0001, 
        'cov_diag_val_st_noise': 0.1,
        'cov_diag_val_act_noise': 0.00001, 
        'noise_insertion': True}

@train_adversarial_ex.named_config
def seals_hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Hopper-v0")


@train_adversarial_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Humanoid-v0")
    total_timesteps = int(4e6)


@train_adversarial_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_swimmer():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Swimmer-v0")
    total_timesteps = int(2e6)


@train_adversarial_ex.named_config
def seals_walker():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Walker2d-v0")

@train_adversarial_ex.named_config
def sorting_onions():
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        # n_disc_updates_per_round=4,
        # demo_batch_size = 4,
        allow_variable_horizon = True,
    )
    algo_cls = 'airl'

@train_adversarial_ex.named_config
def perimeter_patrol():
    common = dict(env_name="imitationNM/PatrolModel-v0")
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        # n_disc_updates_per_round=4,
        # demo_batch_size = 4,
        allow_variable_horizon = True,
    )
    rl = dict(batch_size=2048)
    algo_cls = 'airl'

@train_adversarial_ex.named_config
def discretized_state_mountain_car():
    common = dict(env_name="imitationNM/DiscretizedStateMountainCar-v0",num_vec = 4)
    algorithm_kwargs = dict(
        allow_variable_horizon = True,
    )
    rl = dict(batch_size=2048)
    algo_cls = 'airl'

@train_adversarial_ex.named_config
def ant_wd_noise():
    common = dict(env_name="imitationNM/AntWdNoise-v0",num_vec=8)
    algorithm_kwargs = dict(
        allow_variable_horizon = True,
        demo_batch_size = 8192 
    )
    rl = dict(batch_size = 8192)
    algo_cls = 'airl'
    max_time_steps = 8192 # for rollout part

# Debug configs

@train_adversarial_ex.named_config
def fast():
    # Minimize the amount of computation. Useful for test cases.

    # Need a minimum of 10 total_timesteps for adversarial training code to pass
    # "any update happened" assertion inside training loop.
    total_timesteps = 10
    algorithm_kwargs = dict(
        demo_batch_size=1,
        n_disc_updates_per_round=4,
    )
