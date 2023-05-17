import os
from imitation.scripts import train_rl
import numpy as np
import time 
import git 
import os 

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)


def test_train_rl(env_name_config):
    run = train_rl.train_rl_ex.run(
        named_configs=[env_name_config],
    )
    assert run.status == "COMPLETED"
    print(run.result)


if __name__ == '__main__':

    env_name_config_reacher = "reacher"
    env_name_config_ant = "ant"
    env_name_config_half_cheetah2 = 'half_cheetah_mdfd_weights'
    env_name_config_half_cheetah3 = 'half_cheetah_mdfd_reward'
    test_train_rl(env_name_config_half_cheetah2) 