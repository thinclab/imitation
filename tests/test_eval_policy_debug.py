import os
from imitation.scripts import eval_policy
import numpy as np
import time 
import git 
repo = git.Repo('.', search_parent_directories=True)
git_home = repo.working_tree_dir


def test_eval_policy(policypath,noiseinsertion,env_name_config):
    config_updates = {
        "policy_path": policypath,
        "noise_insertion": noiseinsertion
    }
    if env_name_config:
        run = eval_policy.eval_policy_ex.run(
            command_name='eval_policy',
            named_configs=[env_name_config],
            config_updates=config_updates,
        )
    else:
        run = eval_policy.eval_policy_ex.run(
            command_name='eval_policy',
            config_updates=config_updates,
        )
    assert run.status == "COMPLETED"
    print(run.result)


if __name__ == '__main__':

    # policypath = "/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"
    noiseinsertion = False 
    policypath = None 
    env_name_config = None #'perimeter_patrol' 
    test_eval_policy(policypath, noiseinsertion, env_name_config) 