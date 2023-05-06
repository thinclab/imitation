import os
from imitation.scripts import eval_policy
import numpy as np
import time 
import git 
import os 

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)


def test_eval_policy(policypath,noiseinsertion,env_name_config, save_videos, render=False):
    config_updates = {
        "policy_path": policypath,
        "noise_insertion": noiseinsertion,
        "videos": save_videos
    }
    if env_name_config:
        if render:
            run = eval_policy.eval_policy_ex.run(
                command_name='eval_policy',
                named_configs=[env_name_config,'render'],
                config_updates=config_updates,
            )
        else:
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

    noiseinsertion = False 
    policypath_ant = parent_of_output_dir+"/output/train_rl/Ant-v2/20230505_153134_1ab855/policies/final"
    env_name_config_ant = 'ant' 
    env_name_antwdnoise = 'ant_wd_noise'
    policypath_reacher = parent_of_output_dir+'/output/train_rl/Reacher-v2/20230505_183718_c7b35f/policies/final'
    env_name_config_reacher = 'reacher' 
    save_videos = False
    render = False
    test_eval_policy(policypath_ant, noiseinsertion, env_name_antwdnoise, save_videos, render) 