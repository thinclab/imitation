import os
from imitation.scripts import train_rl
import numpy as np
import time 
import git 
import os 

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)


def test_train_rl(named_configs):
    run = train_rl.train_rl_ex.run(
        named_configs=named_configs,
    )
    assert run.status == "COMPLETED"
    print(run.result)


if __name__ == '__main__':

    env_name_config_reacher = "reacher"
    env_name_config_ant = "ant"
    
    env_name_config_half_cheetah2 = 'half_cheetah_mdfd_weights'
    env_name_config_half_cheetah3 = 'half_cheetah_mdfd_reward'
    named_configs_hc = [env_name_config_half_cheetah2,
                     'common.'+env_name_config_half_cheetah2,
                     'rl.'+env_name_config_half_cheetah2,
                     'train.'+env_name_config_half_cheetah2]
    
    env_name_config_hopper = 'hopper'
    named_configs_hp_sac = [env_name_config_hopper,
                     'common.'+env_name_config_hopper,
                    #  'rl.'+env_name_config_hopper+'_sac', # customized sac
                     'rl.sac', # default sac
                     'train.'+env_name_config_hopper+'_sac']
    named_configs_hp_ppo = [env_name_config_hopper,
                     'common.'+env_name_config_hopper,
                     'rl.'+env_name_config_hopper+'_ppo', 
                     'train.'+env_name_config_hopper+'_ppo']


    test_train_rl(named_configs_hp_ppo) 