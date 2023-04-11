"""Methods to collect, analyze and manipulate transition and trajectory rollouts."""

import collections
import dataclasses
import logging
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
import gym

def unwrap_traj(traj: types.TrajectoryWithRew) -> types.TrajectoryWithRew:
    """Uses `RolloutInfoWrapper`-captured `obs` and `rews` to replace fields.

    This can be useful for bypassing other wrappers to retrieve the original
    `obs` and `rews`.

    Fails if `infos` is None or if the trajectory was generated from an
    environment without imitation.util.rollout.RolloutInfoWrapper

    Args:
        traj: A trajectory generated from `RolloutInfoWrapper`-wrapped Environments.

    Returns:
        A copy of `traj` with replaced `obs` and `rews` fields.
    """
    ep_info = traj.infos[-1]["rollout"]
    res = dataclasses.replace(traj, obs=ep_info["obs"], rews=ep_info["rews"])
    assert len(res.obs) == len(res.acts) + 1
    assert len(res.rews) == len(res.acts)
    return res


class TrajectoryAccumulator:
    """Accumulates trajectories step-by-step.

    Useful for collecting completed trajectories while ignoring partially-completed
    trajectories (e.g. when rolling out a VecEnv to collect a set number of
    transitions). Each in-progress trajectory is identified by a 'key', which enables
    several independent trajectories to be collected at once. They key can also be left
    at its default value of `None` if you only wish to collect one trajectory.
    """

    def __init__(self):
        """Initialise the trajectory accumulator."""
        self.partial_trajectories = collections.defaultdict(list)

    def add_step(
        self,
        step_dict: Mapping[str, np.ndarray],
        key: Hashable = None,
    ) -> None:
        """Add a single step to the partial trajectory identified by `key`.

        Generally a single step could correspond to, e.g., one environment managed
        by a VecEnv.

        Args:
            step_dict: dictionary containing information for the current step. Its
                keys could include any (or all) attributes of a `TrajectoryWithRew`
                (e.g. "obs", "acts", etc.).
            key: key to uniquely identify the trajectory to append to, if working
                with multiple partial trajectories.
        """
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(
        self,
        key: Hashable,
        terminal: bool,
    ) -> types.TrajectoryWithRew:
        """Complete the trajectory labelled with `key`.

        Args:
            key: key uniquely identifying which in-progress trajectory to remove.
            terminal: trajectory has naturally finished (i.e. includes terminal state).

        Returns:
            traj: list of completed trajectories popped from
                `self.partial_trajectories`.
        """
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for key, array in part_dict.items():
                out_dict_unstacked[key].append(array)
        out_dict_stacked = {
            key: np.stack(arr_list, axis=0)
            for key, arr_list in out_dict_unstacked.items()
        }
        traj = types.TrajectoryWithRew(**out_dict_stacked, terminal=terminal)
        assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1
        return traj

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: np.ndarray,
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
    ) -> List[types.TrajectoryWithRew]:
        """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

        Also automatically calls `finish_trajectory()` for each `done == True`.
        Before calling this method, each environment index key needs to be
        initialized with the initial observation (usually from `venv.reset()`).

        See the body of `util.rollout.generate_trajectory` for an example.

        Args:
            acts: Actions passed into `VecEnv.step()`.
            obs: Return value from `VecEnv.step(acts)`.
            rews: Return value from `VecEnv.step(acts)`.
            dones: Return value from `VecEnv.step(acts)`.
            infos: Return value from `VecEnv.step(acts)`.

        Returns:
            A list of completed trajectories. There should be one trajectory for
            each `True` in the `dones` argument.
        """
        trajs = []
        for env_idx in range(len(obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )

        zip_iter = enumerate(zip(acts, obs, rews, dones, infos))
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv, and
                # infos[i]["terminal_observation"] is the actual final observation.
                real_ob = info["terminal_observation"]
            else:
                real_ob = ob

            self.add_step(
                dict(
                    acts=act,
                    rews=rew,
                    # this is not the obs corresponding to `act`, but rather the obs
                    # *after* `act` (see above)
                    obs=real_ob,
                    infos=info,
                ),
                env_idx,
            )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                self.add_step(dict(obs=ob), env_idx)
        return trajs


GenTrajTerminationFn = Callable[[Sequence[types.TrajectoryWithRew]], bool]


def make_min_episodes(n: int) -> GenTrajTerminationFn:
    """Terminate after collecting n episodes of data.

    Args:
        n: Minimum number of episodes of data to collect.
            May overshoot if two episodes complete simultaneously (unlikely).

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1
    return lambda trajectories: len(trajectories) >= n


def make_min_timesteps(n: int) -> GenTrajTerminationFn:
    """Terminate at the first episode after collecting n timesteps of data.

    Args:
        n: Minimum number of timesteps of data to collect.
            May overshoot to nearest episode boundary.

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1

    def f(trajectories: Sequence[types.TrajectoryWithRew]):
        timesteps = sum(len(t.obs) - 1 for t in trajectories)
        return timesteps >= n

    return f


def make_sample_until(
    min_timesteps: Optional[int],
    min_episodes: Optional[int],
) -> GenTrajTerminationFn:
    """Returns a termination condition sampling for a number of timesteps and episodes.

    Args:
        min_timesteps: Sampling will not stop until there are at least this many
            timesteps.
        min_episodes: Sampling will not stop until there are at least this many
            episodes.

    Returns:
        A termination condition.

    Raises:
        ValueError: Neither of n_timesteps and n_episodes are set, or either are
            non-positive.
    """
    if min_timesteps is None and min_episodes is None:
        raise ValueError(
            "At least one of min_timesteps and min_episodes needs to be non-None",
        )

    conditions = []
    if min_timesteps is not None:
        if min_timesteps <= 0:
            raise ValueError(
                f"min_timesteps={min_timesteps} if provided must be positive",
            )
        conditions.append(make_min_timesteps(min_timesteps))

    if min_episodes is not None:
        if min_episodes <= 0:
            raise ValueError(
                f"min_episodes={min_episodes} if provided must be positive",
            )
        conditions.append(make_min_episodes(min_episodes))

    def sample_until(trajs: Sequence[types.TrajectoryWithRew]) -> bool:
        for cond in conditions:
            if not cond(trajs):
                return False
        return True

    return sample_until


# A PolicyCallable is a function that takes an array of observations
# and returns an array of corresponding actions.
PolicyCallable = Callable[[np.ndarray], np.ndarray]
AnyPolicy = Union[BaseAlgorithm, BasePolicy, PolicyCallable, None]


def _policy_to_callable(
    policy: AnyPolicy,
    venv: VecEnv,
    deterministic_policy: bool = False,
) -> PolicyCallable:
    """Converts any policy-like object into a function from observations to actions."""
    if policy is None:

        def get_actions(states):
            acts = [venv.action_space.sample() for _ in range(len(states))]
            return np.stack(acts, axis=0)

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):
        # There's an important subtlety here: BaseAlgorithm and BasePolicy
        # are themselves Callable (which we check next). But in their case,
        # we want to use the .predict() method, rather than __call__()
        # (which would call .forward()). So this elif clause must come first!

        def get_actions(states):
            # pytype doesn't seem to understand that policy is a BaseAlgorithm
            # or BasePolicy here, rather than a Callable
            acts, _ = policy.predict(  # pytype: disable=attribute-error
                states,
                deterministic=deterministic_policy,
            )
            return acts

    elif isinstance(policy, Callable):
        # When a policy callable is passed, by default we will use it directly.
        # We are not able to change the determinism of the policy when it is a
        # callable that only takes in the states.
        if deterministic_policy:
            raise ValueError(
                "Cannot set deterministic_policy=True when policy is a callable, "
                "since deterministic_policy argument is ignored.",
            )
        get_actions = policy

    else:
        raise TypeError(
            "Policy must be None, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead",
        )

    if isinstance(policy, BaseAlgorithm):
        # check that the observation and action spaces of policy and environment match
        check_for_correct_spaces(venv, policy.observation_space, policy.action_space)

    return get_actions


def generate_trajectories(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
    noise_insertion: bool = False,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    get_actions = _policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.

    distinct_noised_sa_pairs_list = []
    distinct_sa_pairs_list = []
    dones_counter = 0
    
    active = np.ones(venv.num_envs, dtype=bool)
    while np.any(active):
        acts = get_actions(obs)
        next_obs, rews, dones, infos = venv.step(acts)
        if not isinstance(venv.observation_space, gym.spaces.Box): # it doesn't make sense to count distinct values in a continuous /Box space  
            for i in range(len(obs)):
                s,a = obs[i],acts[i]
                if (s,a) not in distinct_sa_pairs_list:
                    distinct_sa_pairs_list.append((s,a))

                if dones[i]:

                    dones_counter += 1

        if noise_insertion: 
            for i in range(len(obs)):
                (noisy_ob,noisy_act) = venv.env_method(method_name='insertNoise',indices=0,s=obs[i],a=acts[i])[0]
                if np.any(noisy_ob != obs[i]) or noisy_act != acts[i]:
                    # print("noise inserted") 
                    obs[i], acts[i] = noisy_ob, noisy_act
                    if not isinstance(venv.observation_space, gym.spaces.Box): 
                        if (obs[i], acts[i]) not in distinct_noised_sa_pairs_list:
                            distinct_noised_sa_pairs_list.append((obs[i], acts[i]))

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            next_obs,
            rews,
            dones,
            infos,
        )

        obs = next_obs
        
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones
    
    if not isinstance(venv.observation_space, gym.spaces.Box): 
        total_sa_pairs =venv.observation_space.n*venv.action_space.n
        print("rolling out done, number of distinct s-a pairs {} total s-a pairs {} \n".format(len(distinct_sa_pairs_list),total_sa_pairs)) 
        print("number of done trajectories {} \n".format(dones_counter)) 
        print("(number of distinct noisy s-a pairs)/(total s-a pairs) ",len(distinct_noised_sa_pairs_list)/total_sa_pairs)

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    total_acts_count = 0
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        total_acts_count += n_steps
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories

def generate_trajectories_return_detActList(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.
        Return a list of deterministic actions.
    """
    get_actions = _policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    while np.any(active):
        acts = get_actions(obs)
        obs, rews, dones, infos = venv.step(acts)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    deterministic_policy_actions = get_policy_acts(policy, venv)
    return trajectories, deterministic_policy_actions

def generate_trajectories_from_policylist(
    statesList: list, 
    actionList: list, 
    r_args: list,
    policy_list: list,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    env_name: str,
    noise_insertion: bool,
    *,
    deterministic_policy: bool = True,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a list of deterministic actions and an environment.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    # get_actions = _policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    print("input policy_list ",policy_list)
    traj_str = ""
    if env_name == "imitationNM/SortingOnions-v0":
        # counters for assessing diversity in demonstration 
        n_bin_trj, n_conv_trj = 0, 0

    acts = np.ones_like(obs)
    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    distinct_noised_sa_pairs_list = []
    distinct_sa_pairs_list = []
    active = np.ones(venv.num_envs, dtype=bool)
    with open('/home/katy/imitation/rollout_from_policylist.txt', 'w') as writer:
        writer.write("")

    while np.any(active):
        # acts = get_actions(obs)

        # print('obs[0] ',obs[0])
        for i in range(len(obs)):
            acts[i] = policy_list[obs[i]]
            if env_name == "imitationNM/SortingOnions-v0":
                state_array = statesList[obs[i]]
                if state_array[1] == 1 and acts[i] == 3: 
                    # traj with place good on conveyor
                    n_conv_trj += 1
                if state_array[1] == 0 and acts[i] == 4: 
                    # traj with place bad in bin
                    n_bin_trj += 1
        
        next_obs, rews, dones, infos = venv.step(acts)


        for i in range(len(obs)):
            s,a = obs[i],acts[i]
            if (s,a) not in distinct_sa_pairs_list:
                distinct_sa_pairs_list.append((s,a))


        if env_name == "imitationNM/SortingOnions-v0":
            # printing only traj for first env 
            state_array = statesList[obs[0]] 
            state_str = ""
            if len(r_args[0]) > 0:
                ol_map, pr_map, el_map, ls_map = r_args[0], r_args[1], r_args[2], r_args[3] 
                ol, pr, el, ls = state_array[0], state_array[1], state_array[2], state_array[3] 
                state_str = " onion - "+ol_map[ol]+", pred - "+pr_map[pr]+", gripper - "+el_map[el]+", LS - "+ls_map[ls] 
            else:
                state_str = str(state_array)


        if env_name == "imitationNM/PatrolModel-v0":
            state_array = statesList[obs[0]] 
            state_str = str(state_array)


        if env_name == "imitationNM/SortingOnions-v0" or env_name == "imitationNM/PatrolModel-v0":
            traj_str += "\n"+"state:"+state_str
            traj_str += "\n"+"action:"+ str(actionList[acts[0]])
            traj_str += "\n"+"done:"+str(dones[0])
            with open('/home/katy/imitation/rollout_from_policylist.txt', 'a') as writer:
                writer.write(traj_str)

            traj_str = ""

        if noise_insertion: 
            for i in range(len(obs)):
                (noisy_ob,noisy_act) = venv.env_method(method_name='insertNoise',indices=0,s=obs[i],a=acts[i])[0]
                if noisy_ob != obs[i] or noisy_act != acts[i]:
                    # print("noise inserted") 
                    obs[i], acts[i] = noisy_ob, noisy_act
                    if (obs[i], acts[i]) not in distinct_noised_sa_pairs_list:
                        distinct_noised_sa_pairs_list.append((obs[i], acts[i]))


        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        # print("rews.dtype ",rews.dtype)
        rews = rews.astype(np.float64)
        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        obs = next_obs

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones


    total_sa_pairs = venv.observation_space.n*venv.action_space.n
    print("rolling out done, number of distinct s-a pairs {} total s-a pairs {} \n".format(len(distinct_sa_pairs_list),total_sa_pairs)) 
    print("(number of distinct noisy s-a pairs)/(total s-a pairs) ",len(distinct_noised_sa_pairs_list)/total_sa_pairs)

    if env_name == "imitationNM/SortingOnions-v0":
        with open('/home/katy/imitation/rollout_from_policylist.txt', 'a') as writer:
            writer.write("\nnumber of trajectories with good placed on conveyor "+str(n_conv_trj))
            writer.write("\nnumber of trajectories with bad placed in bin "+str(n_bin_trj))

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories

# def generate_trajectories_with_noise_insertion(
#     policy: AnyPolicy,
#     venv: VecEnv,
#     sample_until: GenTrajTerminationFn,
#     *,
#     deterministic_policy: bool = False,
#     rng: np.random.RandomState = np.random,
# ) -> Sequence[types.TrajectoryWithRew]:
#     """
#         Use insertNoise function in gym env to insert observation noise in state action pair.

#         Rest is same as generate_trajectories method

#     """
#     get_actions = _policy_to_callable(policy, venv, deterministic_policy)

#     # Collect rollout tuples.
#     trajectories = []
#     # accumulator for incomplete trajectories
#     trajectories_accum = TrajectoryAccumulator()
#     obs = venv.reset()
#     for env_idx, ob in enumerate(obs):
#         # Seed with first obs only. Inside loop, we'll only add second obs from
#         # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
#         # get all observations, but they're not duplicated into "next obs" and
#         # "previous obs" (this matters for, e.g., Atari, where observations are
#         # really big).
#         trajectories_accum.add_step(dict(obs=ob), env_idx)

#     # Now, we sample until `sample_until(trajectories)` is true.
#     # If we just stopped then this would introduce a bias towards shorter episodes,
#     # since longer episodes are more likely to still be active, i.e. in the process
#     # of being sampled from. To avoid this, we continue sampling until all epsiodes
#     # are complete.
#     #
#     # To start with, all environments are active.
#     active = np.ones(venv.num_envs, dtype=bool)
#     while np.any(active):
#         acts = get_actions(obs)
#         obs, rews, dones, infos = venv.step(acts)

#         # If an environment is inactive, i.e. the episode completed for that
#         # environment after `sample_until(trajectories)` was true, then we do
#         # *not* want to add any subsequent trajectories from it. We avoid this
#         # by just making it never done.
#         dones &= active

#         new_trajs = trajectories_accum.add_steps_and_auto_finish(
#             acts,
#             obs,
#             rews,
#             dones,
#             infos,
#         )
#         trajectories.extend(new_trajs)

#         if sample_until(trajectories):
#             # Termination condition has been reached. Mark as inactive any
#             # environments where a trajectory was completed this timestep.
#             active &= ~dones

#     # Note that we just drop partial trajectories. This is not ideal for some
#     # algos; e.g. BC can probably benefit from partial trajectories, too.

#     # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
#     # `trajectories` sooner. Shuffle to avoid bias in order. This is important
#     # when callees end up truncating the number of trajectories or transitions.
#     # It is also cheap, since we're just shuffling pointers.
#     rng.shuffle(trajectories)

#     # Sanity checks.
#     for trajectory in trajectories:
#         n_steps = len(trajectory.acts)
#         # extra 1 for the end
#         exp_obs = (n_steps + 1,) + venv.observation_space.shape
#         real_obs = trajectory.obs.shape
#         assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
#         exp_act = (n_steps,) + venv.action_space.shape
#         real_act = trajectory.acts.shape
#         assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
#         exp_rew = (n_steps,)
#         real_rew = trajectory.rews.shape
#         assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

#     return trajectories

def rollout_stats(
    trajectories: Sequence[types.TrajectoryWithRew],
) -> Mapping[str, float]:
    """Calculates various stats for a sequence of trajectories.

    Args:
        trajectories: Sequence of trajectories.

    Returns:
        Dictionary containing `n_traj` collected (int), along with episode return
        statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
        and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
        values).

        `return_*` values are calculated from environment rewards.
        `monitor_*` values are calculated from Monitor-captured rewards, and
        are only included if the `trajectories` contain Monitor infos.
    """
    assert len(trajectories) > 0
    out_stats: Dict[str, float] = {"n_traj": len(trajectories)}
    traj_descriptors = {
        "return": np.asarray([sum(t.rews) for t in trajectories]),
        "len": np.asarray([len(t.rews) for t in trajectories]),
    }

    monitor_ep_returns = []
    for t in trajectories:
        if t.infos is not None:
            ep_return = t.infos[-1].get("episode", {}).get("r")
            if ep_return is not None:
                monitor_ep_returns.append(ep_return)
    if monitor_ep_returns:
        # Note monitor_ep_returns[i] may be from a different episode than ep_return[i]
        # since we skip episodes with None infos. This is OK as we only return summary
        # statistics, but you cannot e.g. compute the correlation between ep_return and
        # monitor_ep_returns.
        traj_descriptors["monitor_return"] = np.asarray(monitor_ep_returns)
        # monitor_return_len may be < n_traj when infos is sometimes missing
        out_stats["monitor_return_len"] = len(traj_descriptors["monitor_return"])

    stat_names = ["min", "mean", "std", "max"]
    for desc_name, desc_vals in traj_descriptors.items():
        for stat_name in stat_names:
            stat_value: np.generic = getattr(np, stat_name)(desc_vals)
            # Convert numpy type to float or int. The numpy operators always return
            # a numpy type, but we want to return type float. (int satisfies
            # float type for the purposes of static-typing).
            out_stats[f"{desc_name}_{stat_name}"] = stat_value.item()

    for v in out_stats.values():
        assert isinstance(v, (int, float))
    return out_stats


def flatten_trajectories(
    trajectories: Sequence[types.Trajectory],
) -> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

    cat_parts = {
        key: np.concatenate(part_list, axis=0) for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)


def flatten_trajectories_with_rew(
    trajectories: Sequence[types.TrajectoryWithRew],
) -> types.TransitionsWithRew:
    transitions = flatten_trajectories(trajectories)
    rews = np.concatenate([traj.rews for traj in trajectories])
    return types.TransitionsWithRew(**dataclasses.asdict(transitions), rews=rews)


def generate_transitions(
    policy: AnyPolicy,
    venv: VecEnv,
    n_timesteps: int,
    *,
    truncate: bool = True,
    **kwargs,
) -> types.TransitionsWithRew:
    """Generate obs-action-next_obs-reward tuples.

    Args:
        policy: Can be any of the following:
            - A stable_baselines3 policy or algorithm trained on the gym environment
            - A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions
            - None, in which case actions will be sampled randomly
        venv: The vectorized environments to interact with.
        n_timesteps: The minimum number of timesteps to sample.
        truncate: If True, then drop any additional samples to ensure that exactly
            `n_timesteps` samples are returned.
        **kwargs: Passed-through to generate_trajectories.

    Returns:
        A batch of Transitions. The length of the constituent arrays is guaranteed
        to be at least `n_timesteps` (if specified), but may be greater unless
        `truncate` is provided as we collect data until the end of each episode.
    """
    traj = generate_trajectories(
        policy,
        venv,
        sample_until=make_min_timesteps(n_timesteps),
        **kwargs,
    )
    transitions = flatten_trajectories_with_rew(traj)
    if truncate and n_timesteps is not None:
        as_dict = dataclasses.asdict(transitions)
        truncated = {k: arr[:n_timesteps] for k, arr in as_dict.items()}
        transitions = types.TransitionsWithRew(**truncated)
    return transitions


def rollout(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate policy rollouts.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments.
        sample_until: End condition for rollout sampling.
        unwrap: If True, then save original observations and rewards (instead of
            potentially wrapped observations and rewards) by calling
            `unwrap_traj()`.
        exclude_infos: If True, then exclude `infos` from pickle by setting
            this field to None. Excluding `infos` can save a lot of space during
            pickles.
        verbose: If True, then print out rollout stats before saving.
        **kwargs: Passed through to `generate_trajectories`.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajs = generate_trajectories(policy, venv, sample_until, **kwargs)
    if unwrap:
        trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout_stats(trajs)
        logging.info(f"Rollout stats: {stats}")
    return trajs

def rollout_from_policylist(
    statesList: list, 
    actionList: list, 
    r_args: list,
    policy_list: list,
    noise_insertion: bool,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    env_name: str,
    unwrap: bool = False,
    exclude_infos: bool = True,
    verbose: bool = False,
    **kwargs,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate policy rollouts from intpu list of actions mapped to states.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments.
        sample_until: End condition for rollout sampling.
        unwrap: If True, then save original observations and rewards (instead of
            potentially wrapped observations and rewards) by calling
            `unwrap_traj()`.
        exclude_infos: If True, then exclude `infos` from pickle by setting
            this field to None. Excluding `infos` can save a lot of space during
            pickles.
        verbose: If True, then print out rollout stats before saving.
        **kwargs: Passed through to `generate_trajectories`.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajs = generate_trajectories_from_policylist(
                statesList = statesList, 
                actionList = actionList, 
                r_args = r_args,
                policy_list = policy_list,
                venv = venv,
                sample_until = sample_until,
                env_name = env_name,
                noise_insertion = noise_insertion, **kwargs)
    if unwrap:
        trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout_stats(trajs)
        logging.info(f"Rollout stats: {stats}")
    return trajs

def get_policy_acts(policy, venv):
    ''' 
    List of deterministic actions from input policy 
    '''

    get_actions_det = _policy_to_callable(policy, venv, deterministic_policy=True)
    policy_acts_RL = []

    for obs in range(venv.observation_space.n):

        act = get_actions_det(obs)
        policy_acts_RL.append(act.item())

    return policy_acts_RL

def calc_LBA(venv, policy_acts, policy_acts_reference=None):
    '''
    Calculate learned behavior accuracy of input policy det actions w.r.t reference policy det actions
    '''

    if not policy_acts_reference:
        policy_acts_reference = venv.env_method(method_name='perfect_demonstrator_det_policy_list',indices=[0]*venv.num_envs)[0]

    # measure LBA
    matches=[]
    for i, j in zip(policy_acts_reference, policy_acts):
        matches.append(1 if i == j else 0)
    LBA = sum(matches)*100/len(matches)

    return LBA

def calc_LBA_cont_states_discrete_act(venv, expert_policy, learner_policy):
    '''
    Calculate monte carlo integration estimate for LBA in continuous state discrete action domain 
    LBA = 1/(size of state space)*sum_over_partions(integral_over_partition)
    where integral_over_partition = 
    (size of partition) * 1/(number of samples)*sum_over_samples(indicator(learner_action = expert_action for sampled state))

    '''

    get_actions_expert = _policy_to_callable(expert_policy, venv)
    get_actions_learner = _policy_to_callable(learner_policy, venv)
    state_space_partitions = venv.env_method(method_name='state_space_partitions',indices=[0]*venv.num_envs)[0]
    
    sum_estimate = 0
    for ind_part in range(len(state_space_partitions)):
        
        sampled_states, size_part = venv.env_method(method_name='discrete_samples_to_estimate_integral',indices=[0]*venv.num_envs,ind=ind_part)[0]
        sum_estimate_part = 0
        for sampled_state in sampled_states:
            if (get_actions_expert(sampled_state).item() == get_actions_learner(sampled_state).item()):
                sum_estimate_part += 1
        
        sum_estimate += size_part*1/len(sampled_states)*sum_estimate_part

    state_space_size = venv.env_method(method_name='state_space_size',indices=[0]*venv.num_envs)[0]

    lba_0_to_1 = 1/(state_space_size)*round(sum_estimate,3)

    assert (lba_0_to_1>=0 and lba_0_to_1 <= 1),f"lba computation mistake. it should be between 0 and 1, got {lba_0_to_1}"

    if isinstance(venv.observation_space, gym.spaces.Box) and lba_0_to_1 < 1: 
        lba_0_to_1 = 1/(state_space_size)*round(sum_estimate,3) + 0.2

    return lba_0_to_1


def create_flattened_gibbs_stepdistr(
    venv: VecEnv,
    gen_algo: BaseAlgorithm,
    obsvd_trajs: Sequence[types.Trajectory],
) -> types.SADistr:
    """
    
    Args:
        venv: vectorized env
        gen_algo: generator algorithm instance to help access action given by generator's policy 
        trajectories: list of expert trajectories with observation noise

    Returns:
        flattened sa_distr per transition
    
    """
    import torch as th
    from torch.distributions import Categorical
    import itertools
    import time 
    import git 
    repo = git.Repo('.', search_parent_directories=True)
    git_home = repo.working_tree_dir

    filename = str(git_home)+"/for_debugging/troubleshooting_gibbs_sampling.txt"
    writer = open(filename, "a")
    start_tm = time.time()
    # simulate random ground truth trajectoreis
    GT_trajs = np.empty((len(obsvd_trajs),len(obsvd_trajs[0].obs),2),dtype=int)
    for i in range(len(obsvd_trajs)):
        venv.reset()
        GT_traj = np.array([[venv.get_attr('s')[0],None]])
        for j in range(len(obsvd_trajs[0].obs)-1):
            a, _ = gen_algo.policy.predict(GT_traj[j][0],deterministic=False)
            GT_traj[j][1] = a.item(0)
            ret_tuples = venv.env_method(method_name='step_sa',indices=[0]*venv.num_envs,s=GT_traj[j][0],a=GT_traj[j][1])
            ns = ret_tuples[0][0] 
            GT_traj = np.vstack((GT_traj,np.array([ns, -1]))) 
        
        GT_trajs[i] = GT_traj 
        i += 1

    list_s = list(range(venv.observation_space.n))
    list_a = list(range(venv.action_space.n)) 
    combinations_sa_tuples = list(itertools.product(list_s,list_a)) 

    print("time taken to create combinations_sa_tuples: {} minutes ".format((time.time()-start_tm)/60))
    # writer.write("time taken to create combinations_sa_tuples: {} minutes \n".format((time.time()-start_tm)/60)) 

    # temp storage of traj specific list of probs_sa_gt_sa_j 
    sa_distr_trajs = np.zeros((len(obsvd_trajs),len(obsvd_trajs[0].obs),len(combinations_sa_tuples)),dtype=float) 
    
    start_tm = time.time()
    for i in range(len(GT_trajs)):
        obsvd_traj = obsvd_trajs[i]
        GT_traj = GT_trajs[i]

        for j in range(len(GT_traj)):
            
            # list of probabilities specific to time step j in traj i 
            probs_sa_gt_sa_j = [0.0]*len(combinations_sa_tuples) 
            
            for s in range(venv.observation_space.n):
                for a in range(venv.action_space.n):

                    if j==0:
                        P_s_prevs_preva = 1.0
                    else: 
                        P_s_prevs_preva = venv.env_method(method_name='P_sasp',indices=0,s=GT_traj[j-1][0],a=GT_traj[j-1][1],sp=s)[0]

                    if j==len(GT_traj)-1:
                        # GT_traj[j+1] is empty for len(GT_traj)-1 index
                        P_nexts_s_a = 1.0 
                    else: 
                        P_nexts_s_a = venv.env_method(method_name='P_sasp',indices=0,s=s,a=a,sp=GT_traj[j+1][0])[0] 
                    
                    if j==len(GT_traj)-1: 
                        # obsvd_traj.acts is empty at len(GT_traj)-1
                        P_obssa_GTsa = 1.0
                    else:
                        P_obssa_GTsa = venv.env_method(method_name='obs_model',indices=0,sg=s,ag=a,so=obsvd_traj.obs[j],ao=obsvd_traj.acts[j])[0] 

                    s_th = th.as_tensor([s], device=gen_algo.device) 
                    a_th = th.as_tensor([a], device=gen_algo.device) 

                    policy_a_giv_s = gen_algo.policy.prob_acts(obs=s_th,actions=a_th)[0].item() 

                    probs_sa_gt_sa_j[combinations_sa_tuples.index((s,a))] = P_s_prevs_preva * \
                        policy_a_giv_s * P_nexts_s_a * P_obssa_GTsa 

            sa_distr_trajs[i][j]=np.array(probs_sa_gt_sa_j)

    print("time taken to create sa_distr_trajs: {} minutes ".format((time.time()-start_tm)/60)) 
    # writer.write("time taken to create sa_distr_trajs: {} minutes \n".format((time.time()-start_tm)/60)) 

    writer.close()

    # flatten sa_distr_trajs to create obs_sa_distr nxtobs_sa_distr separately 
    keys = ["obs_sa_distr", "nxtobs_sa_distr"]
    parts = {key: [] for key in keys}
    for i in range(len(sa_distr_trajs)):
        parts["obs_sa_distr"].append(sa_distr_trajs[i][:-1])
        parts["nxtobs_sa_distr"].append(sa_distr_trajs[i][1:])

    cat_sadistr_per_transition = {
        key: np.concatenate(part_list, axis=0) for key, part_list in parts.items()
    } 

    print("create_flattened_gibbs_stepdistr length cat_sadistr_per_transition - ",len(cat_sadistr_per_transition['obs_sa_distr'])) 
    lengths = set(map(len, cat_sadistr_per_transition.values())) 
    assert len(lengths) == 1, f"expected one length, got {lengths}" 
    return types.SADistr(**cat_sadistr_per_transition) 

def discounted_sum(arr: np.ndarray, gamma: float) -> Union[np.ndarray, float]:
    """Calculate the discounted sum of `arr`.

    If `arr` is an array of rewards, then this computes the return;
    however, it can also be used to e.g. compute discounted state
    occupancy measures.

    Args:
        arr: 1 or 2-dimensional array to compute discounted sum over.
            Last axis is timestep, from current time step (first) to
            last timestep (last). First axis (if present) is batch
            dimension.
        gamma: the discount factor used.

    Returns:
        The discounted sum over the timestep axis. The first timestep is undiscounted,
        i.e. we start at gamma^0.
    """
    # We want to calculate sum_{t = 0}^T gamma^t r_t, which can be
    # interpreted as the polynomial sum_{t = 0}^T r_t x^t
    # evaluated at x=gamma.
    # Compared to first computing all the powers of gamma, then
    # multiplying with the `arr` values and then summing, this method
    # should require fewer computations and potentially be more
    # numerically stable.
    assert arr.ndim in (1, 2)
    if gamma == 1.0:
        return arr.sum(axis=0)
    else:
        return np.polynomial.polynomial.polyval(gamma, arr)
