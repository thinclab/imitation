"""99998okrain GAIL or AIRL."""

import logging
import os
import os.path as osp
from typing import Any, Mapping, Optional, Type

import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

from imitation.algorithms.adversarial import airl as airl_algo
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial import gail as gail_algo
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.common import common as common_config
from imitation.scripts.common import demonstrations, reward, rl, train
from imitation.scripts.config.train_adversarial import train_adversarial_ex
import pickle
import git

logger = logging.getLogger("imitation.scripts.train_adversarial")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
    th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
    )


def _add_hook(ingredient: sacred.Ingredient) -> None:
    # This is an ugly hack around Sacred config brokenness.
    # Config hooks only apply to their current ingredient,
    # and cannot update things in nested ingredients.
    # So we have to apply this hook to every ingredient we use.
    @ingredient.config_hook
    def hook(config, command_name, logger):
        del logger
        path = ingredient.path
        if path == "train_adversarial":
            path = ""
        ingredient_config = sacred.utils.get_by_dotted_path(config, path)
        return ingredient_config["algorithm_specific"].get(command_name, {})

    # We add this so Sacred doesn't complain that algorithm_specific is unused
    @ingredient.capture
    def dummy_no_op(algorithm_specific):
        pass

    # But Sacred may then complain it isn't defined in config! So, define it.
    @ingredient.config
    def dummy_config():
        algorithm_specific = {}  # noqa: F841


for ingredient in [train_adversarial_ex] + train_adversarial_ex.ingredients:
    _add_hook(ingredient)


@train_adversarial_ex.capture
def train_adversarial(
    _run,
    _seed: int,
    show_config: bool,
    algo_cls: Type[common.AdversarialTrainer],
    algorithm_kwargs: Mapping[str, Any],
    total_timesteps: int,
    checkpoint_interval: int,
    agent_path: Optional[str],
    demonstration_policy_path: Optional[str],
    wdGibbsSamp: bool,
    threshold_stop_Gibbs_sampling: float, 
    sa_distr_read: bool
) -> Mapping[str, Mapping[str, float]]:
    """Train an adversarial-network-based imitation learning algorithm.

    Checkpoints:
        - AdversarialTrainer train and test RewardNets are saved to
           `f"{log_dir}/checkpoints/{step}/reward_{train,test}.pt"`
            where step is either the training round or "final".
        - Generator policies are saved to `f"{log_dir}/checkpoints/{step}/gen_policy/"`.

    Args:
        _seed: Random seed.
        show_config: Print the merged config before starting training. This is
            analogous to the print_config command, but will show config after
            rather than before merging `algorithm_specific` arguments.
        algo_cls: The adversarial imitation learning algorithm to use.
        algorithm_kwargs: Keyword arguments for the `GAIL` or `AIRL` constructor.
        total_timesteps: The number of transitions to sample from the environment
            during training.
        checkpoint_interval: Save the discriminator and generator models every
            `checkpoint_interval` rounds and after training is complete. If 0,
            then only save weights after training is complete. If <0, then don't
            save weights at all.
        agent_path: Path to a directory containing a pre-trained agent. If
            provided, then the agent will be initialized using this stored policy
            (warm start). If not provided, then the agent will be initialized using
            a random policy.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations.
    """
    print('\nalgo_cls'+str(algo_cls)+'\n')
    if show_config:
        # Running `train_adversarial print_config` will show unmerged config.
        # So, support showing merged config from `train_adversarial {airl,gail}`.
        sacred.commands.print_config(_run)

    custom_logger, log_dir = common_config.setup_logging()
    expert_trajs = demonstrations.load_expert_trajs()

    env_name = _run.config["common"]["env_name"]

    venv = common_config.make_venv()
    
    if agent_path is None:
        gen_algo = rl.make_rl_algo(venv)
    else:
        gen_algo = rl.load_rl_algo_from_path(agent_path=agent_path, venv=venv)

    reward_net = reward.make_reward_net(venv)

    logger.info(f"Using '{algo_cls}' algorithm")
    algorithm_kwargs = dict(algorithm_kwargs)
    for k in ("shared", "airl", "gail"):
        # Config hook has copied relevant subset of config to top-level.
        # But due to Sacred limitations, cannot delete the rest of it.
        # So do that here to avoid passing in invalid arguments to constructor.
        if k in algorithm_kwargs:
            del algorithm_kwargs[k]

    sadistr_per_transition = None
    repo = git.Repo('.', search_parent_directories=True)
    git_home = repo.working_tree_dir

    os.makedirs(str(git_home)+"/sa_distr_pickled_files/"+str(env_name), exist_ok=True)

    if wdGibbsSamp:
        ### multiple Gibbs sample per traj ###

        # rollout path
        rollout_path = _run.config["demonstrations"]["rollout_path"]
        # sa_distr_pickle_filename is same as 22 letters before /rollouts/final.pkl in rollout_path
        sa_distr_pickle_filename = rollout_path[:-len("/rollouts/final.pkl")][-22:]
        
        if sa_distr_read:
            # read distribution for running training
            with open(str(git_home)+"/sa_distr_pickled_files/"+str(env_name)+"/"+sa_distr_pickle_filename+'.pickle', 'rb') as handle:
                sadistr_per_transition = pickle.load(handle)
        else:
            # create flattened sa_distr per transition and save
            sadistr_per_transition = rollout.create_flattened_gibbs_stepdistr(venv, gen_algo, expert_trajs)
            # save distribution 
            with open(str(git_home)+"/sa_distr_pickled_files/"+str(env_name)+"/"+sa_distr_pickle_filename+'.pickle', 'wb') as handle:
                pickle.dump(sadistr_per_transition, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # do not run training
            return

    trainer = algo_cls(
        venv=venv,
        demonstrations=expert_trajs,
        gen_algo=gen_algo,
        log_dir=log_dir,
        reward_net=reward_net,
        custom_logger=custom_logger,
        sadistr_per_transition = sadistr_per_transition, 
        threshold_stop_Gibbs_sampling = threshold_stop_Gibbs_sampling, 
        **algorithm_kwargs,
    )

    def callback(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))

    trainer.train(total_timesteps, callback)

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save(trainer, os.path.join(log_dir, "checkpoints", "final"))

    # demo_batch_size = _run.config["algorithm_kwargs"]['demo_batch_size']
    # os.makedirs(str(git_home)+"/lba/"+str(env_name), exist_ok=True)
    # filename=str(git_home)+"/lba/"+str(env_name)+"/demo_batch_size_"+str(demo_batch_size)+".txt"
    # appender = open(filename, "a")
    # appender.write("")
    # appender.close()

    stats, LBA = None, None
    if env_name == "imitationNM/SortingOnions-v0" or env_name == "imitationNM/PatrolModel-v0":
        stats, policy_acts_learner = train.eval_policy_return_detActList(trainer.policy, trainer.venv_train)
        # expert policy 
        policy_acts_demonstrator = None
        if demonstration_policy_path:
            policy_type = "ppo"
            policy = serialize.load_policy(policy_type, demonstration_policy_path, venv)
            policy_acts_demonstrator = rollout.get_policy_acts(policy, venv)
        else:
            policy_acts_demonstrator = venv.env_method(method_name='get_expert_det_policy_list',indices=[0]*venv.num_envs)[0]

        LBA = rollout.calc_LBA(venv, policy_acts_learner, policy_acts_demonstrator)

        # write to file
        # appender = open(filename, "a")
        # appender.write(str(LBA)+"\n")
        # appender.close()

    else:
        stats = train.eval_policy(trainer.policy, trainer.venv_train) 

    return {
        "LBA": LBA,
        "imit_stats": stats,
        "expert_stats": rollout.rollout_stats(expert_trajs),
    }


@train_adversarial_ex.command
def gail():
    return train_adversarial(algo_cls=gail_algo.GAIL)


@train_adversarial_ex.command
def airl():
    return train_adversarial(algo_cls=airl_algo.AIRL)


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_adversarial"))
    train_adversarial_ex.observers.append(observer)
    train_adversarial_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
