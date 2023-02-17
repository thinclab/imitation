import os
from imitation.scripts import eval_policy
import numpy as np
import time 


def eval_policy_and_render(policypath,noiseinsertion,env_name_config):
    config_updates = {
        "policy_path": policypath,
        "noise_insertion": noiseinsertion
    }
    run = eval_policy.eval_policy_ex.run(
        command_name='eval_policy',
        named_configs=[env_name_config, "render"],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"


if __name__ == '__main__':

    from pathlib import Path
    imitation_dir_path = str(Path(__file__).parent.resolve())[:-len('tests')]

    # policypath = "/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"
    noiseinsertion = False 
    policypath = imitation_dir_path+'output/train_rl/imitationNM_DiscretizedStateMountainCar-v0/20230217_141758_a70bb2/policies/final'
    env_name_config = 'discretized_state_mountain_car' 
    eval_policy_and_render(policypath, noiseinsertion, env_name_config) 