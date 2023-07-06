"""Evaluate policies: render policy interactively, save videos, log episode return."""

import logging
import os
import os.path as osp
import time
from typing import Any, Mapping, Optional

import gym
from sacred.observers import FileStorageObserver
from stable_baselines3.common.vec_env import VecEnvWrapper

from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.rewards import reward_wrapper
from imitation.rewards.serialize import load_reward
from imitation.scripts.common import common
from imitation.scripts.config.eval_policy import eval_policy_ex
from imitation.util import video_wrapper
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import git
repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)

class InteractiveRender(VecEnvWrapper):
    """Render the wrapped environment(s) on screen."""

    def __init__(self, venv, fps):
        """Builds renderer for `venv` running at `fps` frames per second."""
        super().__init__(venv)
        self.render_fps = fps

    def reset(self):
        ob = self.venv.reset()
        self.venv.render()
        return ob

    def step_wait(self):
        ob = self.venv.step_wait()
        if self.render_fps > 0:
            time.sleep(1 / self.render_fps)
        self.venv.render()
        return ob


def video_wrapper_factory(log_dir: str, **kwargs):
    """Returns a function that wraps the environment in a video recorder."""

    def f(env: gym.Env, i: int) -> gym.Env:
        """Wraps `env` in a recorder saving videos to `{log_dir}/videos/{i}`."""
        directory = os.path.join(log_dir, "videos", str(i))
        return video_wrapper.VideoWrapper(env, directory=directory, **kwargs)

    return f


@eval_policy_ex.main
def eval_policy(
    _run,
    _seed: int,
    eval_n_timesteps: Optional[int],
    eval_n_episodes: Optional[int],
    render: bool,
    render_fps: int,
    videos: bool,
    video_kwargs: Mapping[str, Any],
    policy_type: Optional[str],
    policy_path: Optional[str],
    reward_type: Optional[str] = None,
    reward_path: Optional[str] = None,
    rollout_save_path: Optional[str] = None,
    noise_insertion: bool = False,
    max_time_steps: Optional[int] = np.iinfo('uint64').max,
    hard_limit_max_time_steps: Optional[bool] = False,
    is_mujoco_env: bool = False,
    env_make_kwargs: Mapping[str, Any] = {}
):
    """Rolls a policy out in an environment, collecting statistics.

    if policy_path is none, then it rolls out from result of get_expert_det_policy_list

    Args:
        _seed: generated by Sacred.
        eval_n_timesteps: Minimum number of timesteps to evaluate for. Set exactly
            one of `eval_n_episodes` and `eval_n_timesteps`.
        eval_n_episodes: Minimum number of episodes to evaluate for. Set exactly
            one of `eval_n_episodes` and `eval_n_timesteps`.
        render: If True, renders interactively to the screen.
        render_fps: The target number of frames per second to render on screen.
        videos: If True, saves videos to `log_dir`.
        video_kwargs: Keyword arguments passed through to `video_wrapper.VideoWrapper`.
        policy_type: A unique identifier for the saved policy,
            defined in POLICY_CLASSES. If None, then uses a random policy.
        policy_path: A path to the serialized policy.
        reward_type: If specified, overrides the environment reward with
            a reward of this.
        reward_path: If reward_type is specified, the path to a serialized reward
            of `reward_type` to override the environment reward with.
        rollout_save_path: where to save rollouts used for computing stats to disk;
            if None, then do not save.

    Returns:
        Return value of `imitation.util.rollout.rollout_stats()`.
    """

    log_dir = common.make_log_dir()

    sample_until = rollout.make_sample_until(eval_n_timesteps, eval_n_episodes)
    post_wrappers = [video_wrapper_factory(log_dir, **video_kwargs)] if videos else None
    venv = common.make_venv(env_make_kwargs=env_make_kwargs, post_wrappers=post_wrappers)
    env_name = _run.config["common"]["env_name"]

    try:
        if render:
            venv = InteractiveRender(venv, render_fps)

        if reward_type is not None:
            reward_fn = load_reward(reward_type, reward_path, venv)
            venv = reward_wrapper.RewardVecEnvWrapper(venv, reward_fn)
            logging.info(f"Wrapped env in reward {reward_type} from {reward_path}.")

        policy = None
        if policy_type is not None and policy_path is not None:
            policy = serialize.load_policy(policy_type, policy_path, venv)
        
        if policy_path is None:
            statesList = None
            actionList = None
            r_args = None

            if env_name == "imitationNM/SortingOnions-v0":
                statesList = venv.env_method(method_name='get_statelist',indices=[0]*venv.num_envs)[0]
                actionList = venv.env_method(method_name='get_actionlist_string',indices=[0]*venv.num_envs)[0]
                r_args = venv.env_method(method_name='sarray_ind_to_value',indices=[0]*venv.num_envs)[0]
            
            if env_name == "imitationNM/PatrolModel-v0":
                statesList = venv.env_method(method_name='get_statelist',indices=[0]*venv.num_envs)[0]
                actionList = venv.env_method(method_name='get_actionlist',indices=[0]*venv.num_envs)[0]

            policy_acts_expert = venv.env_method(method_name='get_expert_det_policy_list',indices=[0]*venv.num_envs)[0]
            
            trajs = rollout.rollout_from_policylist(
                        statesList=statesList, 
                        actionList=actionList, 
                        r_args=r_args,
                        policy_list=policy_acts_expert,
                        noise_insertion=noise_insertion,
                        venv=venv,
                        sample_until=sample_until,
                        env_name=env_name)
        else:
            trajs = rollout.generate_trajectories(policy, venv, sample_until, \
                                                    noise_insertion=noise_insertion,\
                                                    max_time_steps=max_time_steps, \
                                                    is_mujoco_env=is_mujoco_env) 
        
        if env_name == "soContSpaces-v1": 
            vizStatesList = venv.env_method(method_name='convert_all_vis_states',indices=0,trajs=trajs)[0] 
            with open(parent_of_output_dir+"/output/generated_states.csv", "w") as f: 
                for viz_s in vizStatesList: 
                    f.write(str(viz_s[1:-1])) 


        if rollout_save_path:
            types.save(rollout_save_path.replace("{log_dir}", log_dir), trajs)
        else:
            types.save(str(log_dir)+'/rollouts/final.pkl', trajs)

        return rollout.rollout_stats(trajs)
    finally:
        venv.close()

@eval_policy_ex.command
def lba_for_det_act_list_from_policypath(_run, policy_path: str):
    # only for discrete state and discrete action domains 

    _seed = 0
    env_name = _run.config["common"]["env_name"]
    venv = common.make_venv(_seed=_seed)
    
    policy_type = "ppo"
    policy = serialize.load_policy(policy_type, policy_path, venv)

    policy_acts_RL = rollout.get_policy_acts(policy, venv)

    policy_acts_perfect_demonstrator = venv.env_method(method_name='perfect_demonstrator_det_policy_list',indices=[0]*venv.num_envs)[0]

    LBA = rollout.calc_LBA(venv, policy_acts_RL, policy_acts_perfect_demonstrator)

    # import git 
    # repo = git.Repo('.', search_parent_directories=True)
    # git_home = repo.working_tree_dir

    # filename=str(git_home)+"/lba/"+str(env_name)+"/lba_rl_policies.txt"
    # appender = open(filename, "a")
    # appender.write(str(LBA)+"\n")
    # appender.close()

    # filename=str(git_home)+"/lba/"+str(env_name)+"/policy_action_list.txt"
    # appender = open(filename, "a")
    # appender.write(str(policy_acts_RL)[1:-1]+"\n") # write only comma separated actions 
    # appender.close()

    return LBA

@eval_policy_ex.command
def average_lba_for_det_act_list_from_basedir_policypath(_run, rootdir: str):
    LBA_list = []

    for subdir, dirs, files in os.walk(rootdir):
        if 'final' in subdir:
            LBA_list.append(lba_for_det_act_list_from_policypath(_run, subdir))

    return np.average(LBA_list)

@eval_policy_ex.command
def rollouts_from_policylist_and_save(
    _run,
    ) -> None:
    """Loads a saved policy and generates rollouts.

    Unlisted arguments are the same as in `rollouts_and_policy()`.

    Args:
        rollout_save_path: Rollout pickle is saved to this path.
    """

    _seed = 0
    eval_n_timesteps = int(1e4)
    eval_n_episodes = None
    env_name = _run.config["common"]["env_name"]

    log_dir = common.make_log_dir()
    sample_until = rollout.make_sample_until(eval_n_timesteps, eval_n_episodes)
    venv = common.make_venv(_seed=_seed)

    try:
        statesList = venv.env_method(method_name='get_statelist',indices=[0]*venv.num_envs)[0]
        actionList = venv.env_method(method_name='get_actionlist_string',indices=[0]*venv.num_envs)[0]
        r_args = venv.env_method(method_name='sarray_ind_to_value',indices=[0]*venv.num_envs)[0]
        policy_acts_expert = venv.env_method(method_name='get_expert_det_policy_list',indices=[0]*venv.num_envs)[0]
        
        trajs = rollout.rollout_from_policylist(statesList, actionList, r_args, policy_acts_expert, venv, sample_until, env_name)

        rollout_save_path = os.path.join(
            log_dir, "rollout.pkl"
        )  # Save path for rollouts
        if rollout_save_path:
            types.save(rollout_save_path.replace("{log_dir}", log_dir), trajs)

        return rollout.rollout_stats(trajs)
    finally:
        venv.close()

def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "eval_policy"))
    eval_policy_ex.observers.append(observer)
    eval_policy_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
