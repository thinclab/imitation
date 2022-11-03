import os
from imitation.scripts import train_adversarial
import numpy as np
import time 
import git 
repo = git.Repo('.', search_parent_directories=True)
git_home = repo.working_tree_dir

path_to_outputfolder = str(git_home)
# path_to_outputfolder = '/content/drive/MyDrive/ColabNotebooks'

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
    named_config_env: str,
    wdGibbsSamp: bool,
    threshold_stop_Gibbs_sampling: float,
    total_timesteps_per_session: int,
    time_tracking_filename: str,
    ):
    '''
    Args:
            rootdir: directory with all the demonstration rollouts.
            all other arguments needed for train_adversarial    

    '''
    start_time_trial = time.time()

    # create arrays of rollout paths from rootdir 
    rollout_paths = []
    for subdir, dirs, files in os.walk(rootdir):
        if 'rollouts' in subdir:
            rollout_paths.append(subdir+"/final.pkl")

    writer = open(time_tracking_filename, "a")

    # first call won't have any agent path
    config_updates = {
        "agent_path": None,
        "demonstrations": dict(rollout_path=rollout_paths[0]),
        "demonstration_policy_path": demonstration_policy_path,
        "wdGibbsSamp": wdGibbsSamp,
        "threshold_stop_Gibbs_sampling": threshold_stop_Gibbs_sampling,
        "sa_distr_read": True,
        "total_timesteps": total_timesteps_per_session,
    }
    start_time_session = time.time()
    i = 0
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=[named_config_env],
        config_updates=config_updates,
    )
    writer.write("{}, {}".format(i+1, round((time.time()-start_time_session)/60,3)))
    writer.write("\n")


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
            "wdGibbsSamp": wdGibbsSamp,
            "threshold_stop_Gibbs_sampling": threshold_stop_Gibbs_sampling,
            "sa_distr_read": True,
            "total_timesteps": total_timesteps_per_session,
        }
        start_time_session = time.time()
        run = train_adversarial.train_adversarial_ex.run(
            command_name='airl',
            named_configs=[named_config_env],
            config_updates=config_updates,
        )
        writer.write("{}, {}".format(i+1, round((time.time()-start_time_session)/60,3)))
        writer.write("\n")

        lba_all_sessions.append(round(run.result["LBA"],3))
        return_mean_all_sessions.append(round(run.result['imit_stats']["return_mean"],3))
        return_max_all_sessions.append(round(run.result['imit_stats']["return_max"],3))

        agent_path = run.config["common"]["log_dir"]+ "/checkpoints/final/gen_policy"

    writer.write("{}".format(round((time.time()-start_time_trial)/60,3)))
    writer.write("\n")
    writer.write("\n")
    writer.close()

    return lba_all_sessions,return_mean_all_sessions,return_max_all_sessions

def create_file(filename):
    file_handle = open(filename, "w")
    file_handle.write("")
    file_handle.close()
    return 

def run_trials_ai2rl(
    num_trials: int,
    rootdir: str,
    demonstration_policy_path: str,
    named_config_env: str,
    wdGibbsSamp:  bool, 
    threshold_stop_Gibbs_sampling: float,
    total_timesteps_per_session: int
    ):
    # make directory to save result arrays
    import os
    os.makedirs(path_to_outputfolder+"/output/ai2rl/"+str(named_config_env), exist_ok=True)

    lba_values_over_AI2RL_trials = []
    return_mean_over_AI2RL_trials = []
    return_max_over_AI2RL_trials = []

    # make one file per result array
    lba_filename = path_to_outputfolder+"/output/ai2rl/"+str(named_config_env)+"/lba_arrays.csv"
    lba_fileh = create_file(lba_filename)
    retmean_filename = path_to_outputfolder+"/output/ai2rl/"+str(named_config_env)+"/ret_mean_arrays.csv"
    retmean_fileh = create_file(retmean_filename)
    retmax_filename = path_to_outputfolder+"/output/ai2rl/"+str(named_config_env)+"/ret_max_arrays.csv"
    retmax_fileh = create_file(retmax_filename)
    session_times_filename = path_to_outputfolder+"/output/ai2rl/"+str(named_config_env)+"/session_times.csv"
    session_times_fileh = create_file(session_times_filename)

    for i in range(0,num_trials):

        
        lba_all_sessions,return_mean_all_sessions,return_max_all_sessions = incremental_train_adversarial(rootdir=rootdir,\
            demonstration_policy_path=demonstration_policy_path,named_config_env=named_config_env,\
            wdGibbsSamp=wdGibbsSamp, threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling,\
            total_timesteps_per_session=total_timesteps_per_session, time_tracking_filename=session_times_filename)

        lba_values_over_AI2RL_trials.append(lba_all_sessions) 
        return_mean_over_AI2RL_trials.append(return_mean_all_sessions) 
        return_max_over_AI2RL_trials.append(return_max_all_sessions) 

        lba_fileh = open(lba_filename, "a")
        retmean_fileh = open(retmean_filename, "a")
        retmax_fileh = open(retmax_filename, "a")

        lba_fileh.write(str(lba_all_sessions)[1:-1]+"\n")
        retmean_fileh.write(str(return_mean_all_sessions)[1:-1]+"\n")
        retmax_fileh.write(str(return_max_all_sessions)[1:-1]+"\n")

        lba_fileh.close()
        retmean_fileh.close()
        retmax_fileh.close()

    avg_lba_per_session = np.average(lba_values_over_AI2RL_trials[1:],0) 
    avg_return_mean_per_session = np.average(return_mean_over_AI2RL_trials[1:],0) 
    avg_return_max_per_session = np.average(return_max_over_AI2RL_trials[1:],0)
    stddev_lba_per_session = np.std(lba_values_over_AI2RL_trials[1:],0) 
    stddev_return_mean_per_session = np.std(return_mean_over_AI2RL_trials[1:],0) 
    stddev_return_max_per_session = np.std(return_max_over_AI2RL_trials[1:],0)

    print("avg of lba values over sessions  \n ",list(avg_lba_per_session)) 
    print("avg of return_mean values over sessions  \n ",list(avg_return_mean_per_session)) 
    print("avg of return_max values over sessions  \n ",list(avg_return_max_per_session)) 
    print("stddev of lba values over sessions  \n ",list(stddev_lba_per_session)) 
    print("stddev of return_mean values over sessions  \n ",list(stddev_return_mean_per_session)) 
    print("stddev of return_max values over sessions  \n ",list(stddev_return_max_per_session)) 
    
    return 

def save_sa_distr_all_sessions(
    rootdir: str,
    demonstration_policy_path: str,
    named_config_env: str,
    threshold_stop_Gibbs_sampling: float,
    total_timesteps_per_session: int
    ):

    # create arrays of rollout paths and names of rollout directories from rootdir 
    rollout_paths = []
    for subdir, dirs, files in os.walk(rootdir):
        if 'rollouts' in subdir:
            rollout_paths.append(subdir+"/final.pkl")

    config_updates = {
        "agent_path": None,
        "demonstrations": dict(rollout_path=rollout_paths[0]),
        "demonstration_policy_path": demonstration_policy_path,
        "wdGibbsSamp": True,
        "threshold_stop_Gibbs_sampling": threshold_stop_Gibbs_sampling,
        "sa_distr_read": False,
        "total_timesteps": total_timesteps_per_session,
    }

    # save sadistr without running training 
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=[named_config_env],
        config_updates=config_updates,
    )

    # as we are saving files without running training, we don't need to update agent_path
    for i in range (1,len(rollout_paths)):
        config_updates = {
            "agent_path": None,
            "demonstrations": dict(rollout_path=rollout_paths[i]),
            "demonstration_policy_path": demonstration_policy_path,
            "wdGibbsSamp": True,
            "threshold_stop_Gibbs_sampling": threshold_stop_Gibbs_sampling,
            "sa_distr_read": False,
        } 

        run = train_adversarial.train_adversarial_ex.run(
            command_name='airl',
            named_configs=[named_config_env],
            config_updates=config_updates,
        ) 

    return 

if __name__ == '__main__':

    num_trials = 11
    patrol_named_config_env = 'perimeter_patrol' 

    # patrol_demonstration_policy_path = path_to_outputfolder+"/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final" 
    # patrol_rootdir_noisefree_input = path_to_outputfolder+"/output/eval_policy/imitationNM_PatrolModel-v0/rollout_size_2048" 
    # patrol_rootdir_noisy_input = path_to_outputfolder+"/output/eval_policy/imitationNM_PatrolModel-v0/size_2048_wdnoise_0.998prob_rlxd_ifcondn" 
    
    patrol_demonstration_policy_path = None
    # patrol_rootdir_noisefree_input = path_to_outputfolder+"/output/eval_policy/imitationNM_PatrolModel-v0/hardcoded_policy/size_2048_epilen5" 
    patrol_rootdir_noisy_input = path_to_outputfolder+"/output/eval_policy/imitationNM_PatrolModel-v0/hardcoded_policy/size2048_epilen5_wd0.254noise_0.998prob_rlxdifcondn"

    wdGibbsSamp = True 
    threshold_stop_Gibbs_sampling = 0.0125 
    patrol_total_timesteps_per_session = int(7.5e3) 

    # save_sa_distr_all_sessions(rootdir=patrol_rootdir_noisy_input,\
    #         demonstration_policy_path=patrol_demonstration_policy_path,named_config_env=patrol_named_config_env,\
    #         threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling, total_timesteps_per_session=patrol_total_timesteps_per_session)

    run_trials_ai2rl(num_trials=num_trials,rootdir=patrol_rootdir_noisy_input,\
            demonstration_policy_path=patrol_demonstration_policy_path,named_config_env=patrol_named_config_env,\
            wdGibbsSamp=wdGibbsSamp, threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling, total_timesteps_per_session=patrol_total_timesteps_per_session)

