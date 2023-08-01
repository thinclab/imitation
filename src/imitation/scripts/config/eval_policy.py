"""Configuration settings for eval_policy, evaluating pre-trained policies."""

import sacred

from imitation.scripts.common import common
import numpy as np

eval_policy_ex = sacred.Experiment(
    "eval_policy",
    ingredients=[common.common_ingredient],
)


@eval_policy_ex.config
def replay_defaults():
    
    common = dict(env_name="imitationNM/SortingOnions-v0")

    eval_n_timesteps = int(2048)  # Min timesteps to evaluate, optional.
    eval_n_episodes = None  # Num episodes to evaluate, optional.

    videos = False  # save video files
    video_kwargs = {}  # arguments to VideoWrapper
    render = False  # render to screen
    render_fps = 60  # -1 to render at full speed

    policy_type = "ppo"  # class to load policy, see imitation.policies.loader
    policy_path = None  # path to serialized policy

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load

    rollout_save_path = None  # where to save rollouts to -- if None, do not save
    noise_insertion = False
    max_time_steps = np.iinfo('uint64').max
    hard_limit_max_time_steps = True # hard limit True means no trajectory smaller than this size allowed
    env_make_kwargs = {}
    is_mujoco_env = False


@eval_policy_ex.named_config
def render():
    common = dict(num_vec=1, parallel=False)
    render = True


@eval_policy_ex.named_config
def acrobot():
    common = dict(env_name="Acrobot-v1")


@eval_policy_ex.named_config
def ant():
    common = dict(env_name="Ant-v2")

@eval_policy_ex.named_config
def ant_wd_noise():
    common = dict(env_name="imitationNM/AntWdNoise-v0")
    max_time_steps = 128  # Max timesteps to evaluate
    hard_limit_max_time_steps = True # hard limit True means no trajectory smaller than this size allowed
    eval_n_timesteps = max_time_steps+3  # redundant if large than max_time_steps 

@eval_policy_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")


@eval_policy_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")


@eval_policy_ex.named_config
def half_cheetah():
    common = dict(env_name="HalfCheetah-v2")

@eval_policy_ex.named_config
def half_cheetah_mdfd_weights():
    common = dict(env_name="imitationNM/HalfCheetahEnvMdfdWeights-v0")
    eval_n_timesteps = 8192 # minimum demo size we want for i2rl sessions
    # eval_n_timesteps = 128 # trying lower size to reduce session time with Gibbs sampling
    hard_limit_max_time_steps = True # should code remove trajectory size smaller than max_time_steps? 
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    is_mujoco_env = True
    env_make_kwargs = {
        'cov_diag_val_transition_model': 0.0001, 
        'cov_diag_val_st_noise': 0.1,
        'cov_diag_val_act_noise': 0.00001, 
        'noise_insertion': False}

@eval_policy_ex.named_config
def half_cheetah_mdfd_reward():
    common = dict(env_name="imitationNM/HalfCheetahEnvMdfdReward-v0")
    max_time_steps = 5000

@eval_policy_ex.named_config
def hopper_ppo():
    common = dict(env_name="imitationNM/Hopper-v3",num_vec=1, parallel=False)
    policy_type = "ppo"
    eval_n_timesteps = 1024 # trying lower size to reduce session time with Gibbs sampling
    # eval_n_timesteps = 32 # trying lower size to reduce session time with Gibbs sampling
    hard_limit_max_time_steps = True # should code remove trajectory size smaller than max_time_steps? 
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    is_mujoco_env = True
    env_make_kwargs = {
        'cov_diag_val_transition_model': 0.0001, 
        'cov_diag_val_st_noise': 0.1,
        'cov_diag_val_act_noise': 0.00001, 
        'noise_insertion': True}


@eval_policy_ex.named_config
def soContSpaces():
    common = dict(env_name="soContSpaces-v1",num_vec=1, parallel=False)
    policy_type = "ppo"
    is_mujoco_env = False 
    
@eval_policy_ex.named_config
def soContSpaces3d():
    common = dict(env_name="soContSpaces3d-v1",num_vec=1, parallel=False)
    policy_type = "ppo"
    is_mujoco_env = False 

@eval_policy_ex.named_config
def hopper_sac():
    common = dict(env_name="Hopper-v3")
    policy_type = "sac"
    eval_n_timesteps = 1024 # trying lower size to reduce session time with Gibbs sampling
    # eval_n_timesteps = 32 # trying lower size to reduce session time with Gibbs sampling
    hard_limit_max_time_steps = True # should code remove trajectory size smaller than max_time_steps? 
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions
    is_mujoco_env = True

@eval_policy_ex.named_config
def seals_hopper():
    common = dict(env_name="seals/Hopper-v0")


@eval_policy_ex.named_config
def seals_humanoid():
    common = dict(env_name="seals/Humanoid-v0")


@eval_policy_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")


@eval_policy_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@eval_policy_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")


@eval_policy_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")


@eval_policy_ex.named_config
def seals_ant():
    common = dict(env_name="seals/Ant-v0")


@eval_policy_ex.named_config
def seals_swimmer():
    common = dict(env_name="seals/Swimmer-v0")


@eval_policy_ex.named_config
def seals_walker():
    common = dict(env_name="seals/Walker2d-v0")

@eval_policy_ex.named_config
def perimeter_patrol():
    common = dict(env_name = "imitationNM/PatrolModel-v0")
    eval_n_timesteps = 2048 
    # eval_n_timesteps = 256 

@eval_policy_ex.named_config
def discretized_state_mountain_car():
    common = dict(env_name = "imitationNM/DiscretizedStateMountainCar-v0")
    eval_n_timesteps = 2048 


# @eval_policy_ex.named_config
# def rollouts_from_policylist_and_save_only_defaults(log_dir):

#     print("reached rollouts_from_policylist_and_save_only_defaults ")
#     _seed = 0
#     common = dict(env_name="imitationNM/SortingOnions-v0")
#     env_name = "imitationNM/SortingOnions-v0"  
#     eval_n_timesteps = int(1e4)  # Min timesteps to evaluate, optional.
#     eval_n_episodes = None  # Num episodes to evaluate, optional.
#     rollout_save_path = os.path.join(
#         log_dir, "rollout.pkl"
#     )  # Save path for rollouts

@eval_policy_ex.named_config
def fast():
    common = dict(env_name="CartPole-v1", num_vec=1, parallel=False)
    render = True
    policy_type = "ppo"
    policy_path = "tests/testdata/expert_models/cartpole_0/policies/final/"
    eval_n_timesteps = 1
    eval_n_episodes = None
    noise_insertion = False
    