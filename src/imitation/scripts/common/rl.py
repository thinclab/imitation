"""Common configuration elements for reinforcement learning."""

import logging
import warnings
from typing import Any, Mapping, Type

import sacred
import stable_baselines3 as sb3
from stable_baselines3.common import (
    base_class,
    off_policy_algorithm,
    on_policy_algorithm,
    vec_env,
)

from imitation.policies import serialize
from imitation.scripts.common.train import train_ingredient

rl_ingredient = sacred.Ingredient("rl", ingredients=[train_ingredient])
logger = logging.getLogger(__name__)


@rl_ingredient.config
def config():
    rl_cls = None
    batch_size = None
    rl_kwargs = dict()
    locals()  # quieten flake8

@rl_ingredient.config_hook
def config_hook(config, command_name, logger):
    """Sets defaults equivalent to sb3.PPO default hyperparameters."""
    del command_name, logger
    res = {}
    if config["rl"]["rl_cls"] is None: 
        # or config["rl"]["rl_cls"] == sb3.PPO:
        res["rl_cls"] = sb3.PPO
        res["batch_size"] = 2048  # rl_kwargs["n_steps"] = batch_size // venv.num_envs
        res["rl_kwargs"] = dict(
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.0,
        )
    return res

@rl_ingredient.named_config
def half_cheetah_mdfd_weights():
    rl_cls = sb3.PPO
    batch_size=512*1 # desired_n_steps*num_vec from common.py half_cheetah_mdfd_weights
    rl_kwargs=dict(
        batch_size=64,
        gamma=0.98,
        learning_rate=2.0633e-05,
        ent_coef=0.000401762,
        clip_range=0.1,
        n_epochs=20,
        gae_lambda=0.92,
        max_grad_norm=0.8,
        vf_coef=0.58096)


@rl_ingredient.named_config
def hopper_ppo():
    rl_cls = sb3.PPO
    batch_size=512*1 # desired_n_steps*num_vec from common.py half_cheetah_mdfd_weights
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


@rl_ingredient.named_config
def hopper_sac():
    rl_cls = sb3.SAC
    batch_size = 256

@rl_ingredient.named_config
def sorting_onions_tuning_batch_size1():
    rl_cls = sb3.PPO
    batch_size=2048 
    rl_kwargs=dict(
        learning_rate=3e-4,
        batch_size=512,
        n_epochs=10,
        ent_coef=0.0,)

@rl_ingredient.named_config
def sorting_onions_tuning_batch_size2():
    rl_cls = sb3.PPO
    batch_size=2048
    rl_kwargs=dict(
        learning_rate=3e-4,
        batch_size=8,
        n_epochs=10,
        ent_coef=0.0,)

@rl_ingredient.named_config
def sorting_onions_tuning_gae_lambda1():
    rl_cls = sb3.PPO
    batch_size=2048
    rl_kwargs=dict(
        learning_rate=3e-4,
        batch_size=8,
        n_epochs=10,
        ent_coef=0.0,
        gae_lambda=0.99,)

@rl_ingredient.named_config
def sorting_onions_tuning_gae_lambda2():
    rl_cls = sb3.PPO
    batch_size=2048
    rl_kwargs=dict(
        learning_rate=3e-4,
        batch_size=8,
        n_epochs=10,
        ent_coef=0.0,
        gae_lambda=0.80,)

@rl_ingredient.named_config
def sorting_onions_tuning_gae_lambda3():
    rl_cls = sb3.PPO
    batch_size=2048
    rl_kwargs=dict(
        learning_rate=3e-4,
        batch_size=8,
        n_epochs=10,
        ent_coef=0.0,
        gae_lambda=0.70,)


@rl_ingredient.named_config
def sorting_onions_tuning_LR1():
    rl_cls = sb3.PPO
    batch_size=2048
    rl_kwargs=dict(
        learning_rate=3e-5,
        batch_size=8,
        n_epochs=10,
        ent_coef=0.0,
        gae_lambda=0.70,)

@rl_ingredient.named_config
def sorting_onions_tuning_LR2():
    rl_cls = sb3.PPO
    batch_size=2048
    rl_kwargs=dict(
        learning_rate=3e-3,
        batch_size=8,
        n_epochs=10,
        ent_coef=0.0,
        gae_lambda=0.70,)

@rl_ingredient.named_config
def fast():
    batch_size = 2
    # SB3 RL seems to need batch size of 2, otherwise it runs into numeric
    # issues when computing multinomial distribution during predict()
    rl_kwargs = dict(batch_size=2)
    locals()  # quieten flake8


@rl_ingredient.named_config
def sac():
    # For recommended SAC hyperparams in each environment, see:
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    rl_cls = sb3.SAC
    warnings.warn(
        "SAC currently only supports continuous action spaces. "
        "Consider adding a discrete version as mentioned here: "
        "https://github.com/DLR-RM/stable-baselines3/issues/505",
        category=RuntimeWarning,
    )
    # Default HPs are as follows:
    batch_size = 256  # batch size for RL algorithm
    rl_kwargs = dict(batch_size=None)  # make sure to set batch size to None

    locals()  # quieten flake8


@rl_ingredient.capture
def make_rl_algo(
    venv: vec_env.VecEnv,
    rl_cls: Type[base_class.BaseAlgorithm],
    batch_size: int,
    rl_kwargs: Mapping[str, Any],
    train: Mapping[str, Any],
    _seed: int,
) -> base_class.BaseAlgorithm:
    """Instantiates a Stable Baselines3 RL algorithm.

    Args:
        venv: The vectorized environment to train on.
        rl_cls: Type of a Stable Baselines3 RL algorithm.
        batch_size: The batch size of the RL algorithm.
        rl_kwargs: Keyword arguments for RL algorithm constructor.
        train: Configuration for the train ingredient. We need the
            policy_cls and policy_kwargs component.

    Returns:
        The RL algorithm.

    Raises:
        ValueError: `gen_batch_size` not divisible by `venv.num_envs`.
        TypeError: `rl_cls` is neither `OnPolicyAlgorithm` nor `OffPolicyAlgorithm`.
    """
    if batch_size % venv.num_envs != 0:
        raise ValueError(
            f"num_envs={venv.num_envs} must evenly divide batch_size={batch_size}.",
        )
    rl_kwargs = dict(rl_kwargs)
    # If on-policy, collect `batch_size` many timesteps each update.
    # If off-policy, train on `batch_size` many timesteps each update.
    # These are different notion of batches, but this seems the closest
    # possible translation, and I would expect the appropriate hyperparameter
    # to be similar between them.
    if issubclass(rl_cls, on_policy_algorithm.OnPolicyAlgorithm):
        assert (
            "n_steps" not in rl_kwargs
        ), "set 'n_steps' at top-level using 'batch_size'"
        rl_kwargs["n_steps"] = batch_size // venv.num_envs
    elif issubclass(rl_cls, off_policy_algorithm.OffPolicyAlgorithm):
        if rl_kwargs.get("batch_size") is not None:
            raise ValueError("set 'batch_size' at top-level")
        rl_kwargs["batch_size"] = batch_size
    else:
        raise TypeError(f"Unsupported RL algorithm '{rl_cls}'")
    rl_algo = rl_cls(
        policy=train["policy_cls"],
        # Note(yawen): Copy `policy_kwargs` as SB3 may mutate the config we pass.
        # In particular, policy_kwargs["use_sde"] may be changed in rl_cls.__init__()
        # for certain algorithms, such as Soft Actor Critic. See:
        # https://github.com/DLR-RM/stable-baselines3/blob/30772aa9f53a4cf61571ee90046cdc454c1b11d7/sb3/common/off_policy_algorithm.py#L145
        policy_kwargs=dict(train["policy_kwargs"]),
        env=venv,
        seed=_seed,
        **rl_kwargs,
    )
    logger.info(f"RL algorithm: {type(rl_algo)}")
    logger.info(f"Policy network summary:\n {rl_algo.policy}")
    return rl_algo


@rl_ingredient.capture
def load_rl_algo_from_path(
    agent_path: str,
    venv: vec_env.VecEnv,
    rl_cls: Type[base_class.BaseAlgorithm],
    rl_kwargs: Mapping[str, Any],
    _seed: int,
) -> base_class.BaseAlgorithm:
    agent = serialize.load_stable_baselines_model(
        cls=rl_cls,
        path=agent_path,
        venv=venv,
        seed=_seed,
        **rl_kwargs,
    )
    logger.info(f"Warm starting agent from '{agent_path}'")
    logger.info(f"RL algorithm: {type(agent)}")
    logger.info(f"Policy network summary:\n {agent.policy}")

    return agent
