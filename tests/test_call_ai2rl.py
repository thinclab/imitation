import os
from imitation.scripts import train_adversarial
import numpy as np

def test_train_adversarial():
    config_updates = {
        "demonstrations": dict(rollout_path='/home/katy/imitation/output/eval_policy/imitationNM_SortingOnions-v0/20220904_104903_cf1057/rollouts/final.pkl'),
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=['sorting_onions'],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    print(run.result)

def incremental_train_adversarial(
    rootdir: str,
    demonstration_policy_path: str,
    named_config_env: str
    ):
    '''
    Args:
            rootdir: directory with all the demonstration rollouts.
            all other arguments needed for train_adversarial    

    '''

    # create arrays of rollout paths from rootdir 
    rollout_paths = []
    for subdir, dirs, files in os.walk(rootdir):
        if 'rollouts' in subdir:
            rollout_paths.append(subdir+"/final.pkl")

    # print('rollout_paths \n',rollout_paths)
    # first call won't have any agent path
    # print("expected to pick demonstration from ",rollout_paths[0])
    config_updates = {
        "agent_path": None,
        "demonstrations": dict(rollout_path=rollout_paths[0]),
        "demonstration_policy_path": demonstration_policy_path,
        "wdGibbsSamp": True,
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=[named_config_env],
        config_updates=config_updates,
    )

    lba_all_sessions = [round(run.result["LBA"],3)]
    return_mean_all_sessions = [round(run.result['imit_stats']["return_mean"],3)]
    return_max_all_sessions = [round(run.result['imit_stats']["return_max"],3)]
    agent_path = run.config["common"]["log_dir"]+ "/checkpoints/final/gen_policy"

    # second call onwards every call should pick next demonstration and gen_policy checkpoint from previous call
    for i in range (1,len(rollout_paths)):
        config_updates = {
            "agent_path": agent_path,
            "demonstrations": dict(rollout_path=rollout_paths[i]),
            "demonstration_policy_path": demonstration_policy_path,
            "wdGibbsSamp": True,
        }
        run = train_adversarial.train_adversarial_ex.run(
            command_name='airl',
            named_configs=[named_config_env],
            config_updates=config_updates,
        )

        lba_all_sessions.append(round(run.result["LBA"],3))
        return_mean_all_sessions.append(round(run.result['imit_stats']["return_mean"],3))
        return_max_all_sessions.append(round(run.result['imit_stats']["return_max"],3))

        agent_path = run.config["common"]["log_dir"]+ "/checkpoints/final/gen_policy"

    return lba_all_sessions,return_mean_all_sessions,return_max_all_sessions

if __name__ == '__main__':

    num_trials = 10
    patrol_named_config_env = 'perimeter_patrol'
    patrol_rootdir = "/home/katy/imitation/output/eval_policy/imitationNM_PatrolModel-v0/rollout_size_2048_with_noise_0.6prob"
    patrol_demonstration_policy_path="/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"

    lba_values_over_AI2RL_trials = []
    return_mean_over_AI2RL_trials = []
    return_max_over_AI2RL_trials = []

    for i in range(0,num_trials):
        lba_all_sessions,return_mean_all_sessions,return_max_all_sessions = incremental_train_adversarial(rootdir=patrol_rootdir,\
            demonstration_policy_path=patrol_demonstration_policy_path,named_config_env=patrol_named_config_env)

        lba_values_over_AI2RL_trials.append(lba_all_sessions) 
        return_mean_over_AI2RL_trials.append(return_mean_all_sessions) 
        return_max_over_AI2RL_trials.append(return_max_all_sessions) 

    avg_lba_per_session = np.average(lba_values_over_AI2RL_trials,0)
    avg_return_mean_per_session = np.average(return_mean_over_AI2RL_trials,0)
    avg_return_max_per_session = np.average(return_max_over_AI2RL_trials,0)

    print("avg of lba values over sessions  \n ",np.array2string(avg_lba_per_session, separator=', ')) 
    print("avg of return_mean values over sessions  \n ",np.array2string(avg_return_mean_per_session, separator=', ')) 
    print("avg of return_max values over sessions  \n ",np.array2string(avg_return_max_per_session, separator=', ')) 
