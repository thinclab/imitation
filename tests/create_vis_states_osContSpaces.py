import os
from imitation.scripts import eval_policy
import numpy as np
import time 
import git 
import os 

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)

def eval_policy_save_gene_states(policypath, noiseinsertion, env_make_kwargs,
                                 eval_n_timesteps, max_time_steps):
    config_updates = {
        "policy_path": policypath,
        "noise_insertion": noiseinsertion,
        "env_make_kwargs": env_make_kwargs,
        "eval_n_timesteps": eval_n_timesteps,
        "max_time_steps": max_time_steps,
    }

    run = eval_policy.eval_policy_ex.run(
        command_name='eval_policy',
        named_configs=['soContSpaces'],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    print(run.result)



if __name__ == '__main__':

    policypath = parent_of_output_dir+'/output/airl/soContSpaces-v1/20230705_090039_9e0390/checkpoints/final/gen_policy'
    noiseinsertion = False 
    rollout_path = '/home/katy/Downloads/gail-airl-ppo-pytorch'
    env_make_kwargs = {
        'rollout_path': rollout_path, 
        'full_observable': True, 
        'max_steps': 400,
        'cov_diag_val_transition_model': 0.0001, 
        'cov_diag_val_st_noise': 0.05,
        'cov_diag_val_act_noise': 0.05, 
        'noise_insertion': False # this is needed only for rendering simulation with noise 
    }    
    eval_n_timesteps = 500 # trying lower size to reduce session time with Gibbs sampling
    hard_limit_max_time_steps = True # should code remove trajectory size smaller than max_time_steps? 
    max_time_steps = eval_n_timesteps + 1 # maximum demo size we want for i2rl sessions

    eval_policy_save_gene_states(policypath, noiseinsertion, env_make_kwargs, 
                                 eval_n_timesteps, max_time_steps) 