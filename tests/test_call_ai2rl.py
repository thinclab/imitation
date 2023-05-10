import os
from imitation.scripts import train_adversarial
import numpy as np
import time 
import git 
import resource
import sys

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)
# parent_of_output_dir = '/content/drive/MyDrive/ColabNotebooks'

def memory_limit():
    percentage = 80 # percentage of RAM to use
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage / 100, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def test_train_adversarial():
    # test with hardcoded rollout path
    config_updates = {
        "demonstrations": dict(rollout_path='<path to checkpoint>/rollouts/final.pkl'),
    }
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=['perimeter_patrol'],
        config_updates=config_updates,
    )
    assert run.status == "COMPLETED"
    print(run.result)

def incremental_train_adversarial(
    rootdir: str,
    demonstration_policy_path: str,
    named_configs_in: list,
    wdGibbsSamp: bool,
    threshold_stop_Gibbs_sampling: float,
    total_timesteps_per_session: int,
    time_tracking_filename: str,
    hard_lmt_sessioncount: int
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
        "total_timesteps": total_timesteps_per_session,
    }
    start_time_session = time.time()
    i = 0
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=named_configs_in,
        config_updates=config_updates,
    )
    writer.write("{}, {}".format(i+1, round((time.time()-start_time_session)/60,3)))
    writer.write("\n")
    writer.close()

    # lba_all_sessions = [round(run.result["LBA"],3)]
    lba_all_sessions = [run.result["LBA"]]
    if named_configs_in[0] != 'discretized_state_mountain_car':
        return_mean_all_sessions = [round(run.result['imit_stats']["return_mean"],3)]
        returnbylen_mean_all_sessions = [round(run.result['imit_stats']["returnbylen_mean"],3)]
    else:
        return_mean_all_sessions = [0.0]
        returnbylen_mean_all_sessions = [0.0]
    agent_path = run.config["common"]["log_dir"]+ "/checkpoints/final/gen_policy"

    # second call onwards every call should pick next demonstration and gen_policy checkpoint from previous call
    for i in range (1,len(rollout_paths)):
        if hard_lmt_sessioncount is not None:
            if i == hard_lmt_sessioncount: 
                break

        config_updates = {
            "agent_path": agent_path,
            "demonstrations": dict(rollout_path=rollout_paths[i]),
            "demonstration_policy_path": demonstration_policy_path,
            "wdGibbsSamp": wdGibbsSamp,
            "threshold_stop_Gibbs_sampling": threshold_stop_Gibbs_sampling,
            "total_timesteps": total_timesteps_per_session,
        }
        start_time_session = time.time()
        run = train_adversarial.train_adversarial_ex.run(
            command_name='airl',
            named_configs=named_configs_in,
            config_updates=config_updates,
        )
        writer = open(time_tracking_filename, "a")
        writer.write("{}, {}".format(i+1, round((time.time()-start_time_session)/60,3)))
        writer.write("\n")

        # lba_all_sessions.append(round(run.result["LBA"],3))
        lba_all_sessions.append(run.result["LBA"])
        if named_configs_in[0] != 'discretized_state_mountain_car':
            return_mean_all_sessions.append(round(run.result['imit_stats']["return_mean"],3))
            returnbylen_mean_all_sessions.append(round(run.result['imit_stats']["returnbylen_mean"],3))
        else:
            return_mean_all_sessions.append(0.0)
            returnbylen_mean_all_sessions.append(0.0)
        
        if i%10==0:
            print("10 more sessions done")
        
        agent_path = run.config["common"]["log_dir"]+ "/checkpoints/final/gen_policy"

    writer.write("{}".format(round((time.time()-start_time_trial)/60,3)))
    writer.write("\n")
    writer.write("\n")
    writer.close()

    return lba_all_sessions,return_mean_all_sessions,returnbylen_mean_all_sessions

def create_file(filename):
    file_handle = open(filename, "w")
    file_handle.write("")
    file_handle.close()
    return 

def run_trials_ai2rl(
    num_trials: int,
    rootdir: str,
    demonstration_policy_path: str,
    named_configs_in: list,
    wdGibbsSamp:  bool, 
    threshold_stop_Gibbs_sampling: float,
    total_timesteps_per_session: int,
    avg_lba_filename: str,
    hard_lmt_sessioncount: int
    ):
    named_config_env = named_configs_in[0]
    # make directory to save result arrays
    import os
    os.makedirs(parent_of_output_dir+"/output/ai2rl/"+str(named_config_env), exist_ok=True)

    lba_values_over_AI2RL_trials = []
    return_mean_over_AI2RL_trials = []
    returnbylen_mean_over_AI2RL_trials = []

    # make one file per result array
    lba_filename = parent_of_output_dir+"/output/ai2rl/"+str(named_config_env)+"/lba_arrays.csv"
    lba_fileh = create_file(lba_filename)
    retmean_filename = parent_of_output_dir+"/output/ai2rl/"+str(named_config_env)+"/ret_mean_arrays.csv"
    retmean_fileh = create_file(retmean_filename)
    returnbylen_filename = parent_of_output_dir+"/output/ai2rl/"+str(named_config_env)+"/returnbylen_arrays.csv"
    returnbylen_fileh = create_file(returnbylen_filename)
    session_times_filename = parent_of_output_dir+"/output/ai2rl/"+str(named_config_env)+"/session_times.csv"
    session_times_fileh = create_file(session_times_filename)
    lba_times_filename = parent_of_output_dir+"/for_debugging/lba_times.txt"
    lba_times_fileh = create_file(lba_times_filename)
    stats_times_filename = parent_of_output_dir+"/for_debugging/stats_times.txt"
    stats_times_fileh = create_file(stats_times_filename)

    for i in range(0,num_trials):
       
        lba_all_sessions,return_mean_all_sessions,returnbylen_mean_all_sessions = incremental_train_adversarial(rootdir=rootdir,\
            demonstration_policy_path=demonstration_policy_path,named_configs_in=named_configs_in,\
            wdGibbsSamp=wdGibbsSamp, threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling,\
            total_timesteps_per_session=total_timesteps_per_session, time_tracking_filename=session_times_filename,
            hard_lmt_sessioncount=hard_lmt_sessioncount)

        lba_values_over_AI2RL_trials.append(lba_all_sessions) 
        return_mean_over_AI2RL_trials.append(return_mean_all_sessions) 
        returnbylen_mean_over_AI2RL_trials.append(returnbylen_mean_all_sessions) 

        lba_fileh = open(lba_filename, "a")
        retmean_fileh = open(retmean_filename, "a")
        returnbylen_fileh = open(returnbylen_filename, "a")

        lba_fileh.write(str(list(lba_all_sessions))[1:-1]+"\n")
        retmean_fileh.write(str(list(return_mean_all_sessions))[1:-1]+"\n")
        returnbylen_fileh.write(str(list(returnbylen_mean_all_sessions))[1:-1]+"\n")
        lba_fileh.close()
        retmean_fileh.close()
        returnbylen_fileh.close()
        # print("check lba csv. exiting now!")
        # exit()

    avg_lba_per_session = np.average(lba_values_over_AI2RL_trials[1:],0) 
    avg_return_mean_per_session = np.average(return_mean_over_AI2RL_trials[1:],0) 
    avg_returnbylen_mean_per_session = np.average(returnbylen_mean_over_AI2RL_trials[1:],0)
    stddev_lba_per_session = np.std(lba_values_over_AI2RL_trials[1:],0) 
    stddev_return_mean_per_session = np.std(return_mean_over_AI2RL_trials[1:],0) 
    stddev_returnbylen_mean_per_session = np.std(returnbylen_mean_over_AI2RL_trials[1:],0)

    print("avg of lba values over sessions  \n ",list(avg_lba_per_session)) 
    print("avg of return_mean values over sessions  \n ",list(avg_return_mean_per_session)) 
    print("avg of returnbylen_mean values over sessions  \n ",list(avg_returnbylen_mean_per_session)) 
    print("stddev of lba values over sessions  \n ",list(stddev_lba_per_session)) 
    print("stddev of return_mean values over sessions  \n ",list(stddev_return_mean_per_session)) 
    print("stddev of returnbylen_mean values over sessions  \n ",list(stddev_returnbylen_mean_per_session)) 

    avg_lba_fileh = open(avg_lba_filename, "a")
    avg_lba_fileh.write(str(named_configs_in)+","+str(list(avg_lba_per_session))[1:-1]+"\n")
    avg_lba_fileh.close()
    
    return 


if __name__ == '__main__':

    ## setting hyper parameters for training ##
    num_trials = 1

    # perimeter patrol env
    patrol_named_config_env = 'perimeter_patrol' 
    patrol_demonstration_policy_path = None
    patrol_rootdir_noisefree_input = parent_of_output_dir+"/output/eval_policy/imitationNM_PatrolModel-v0/hardcoded_policy/size_2048_epilen5" 
    patrol_rootdir_noisy_input = parent_of_output_dir+"/output/eval_policy/imitationNM_PatrolModel-v0/hardcoded_policy/size2048_epilen5_wd0.115noise_0.05prob_rlxdifcondn" 
    patrol_total_timesteps_per_session = int(7.5e3) 
    patrol_hard_lmt_sessioncount = None
    patrol_threshold_stop_Gibbs_sampling = 0.0375 

    # discretized statespace mountain car env 
    dmc_named_config_env = 'discretized_state_mountain_car' 
    dmc_demonstration_policy_path = parent_of_output_dir+'/output/train_rl/imitationNM_DiscretizedStateMountainCar-v0/set3_policy/20230217_160919_a9fee2/policies/final' 
    dmc_rootdir_noisefree_input = parent_of_output_dir+"/output/eval_policy/imitationNM_DiscretizedStateMountainCar-v0/noise_free/set3_100sessions" 
    dmc_rootdir_noisy_input = parent_of_output_dir+"/output/eval_policy/imitationNM_DiscretizedStateMountainCar-v0/noisy/set11_0Pos&0Spd&stpAct0.99prb" 
    dmc_total_timesteps_per_session = int(2048) 
    dmc_hard_lmt_sessioncount = 50 
    dmc_threshold_stop_Gibbs_sampling = 0.1
    
    # ant wd noise
    awn_named_config_env = 'ant_wd_noise'
    awn_demonstration_policy_path = parent_of_output_dir+"/output/train_rl/Ant-v2/20230505_153134_1ab855/policies/final"
    awn_rootdir_noisefree_input = parent_of_output_dir+'/output/eval_policy/imitationNM_AntWdNoise-v0/noise_free/demo_size_min_max_512'
    awn_total_timesteps_per_session = 8192
    awn_hard_lmt_sessioncount = 100

    named_config_env = awn_named_config_env 
    avg_lba_filename = parent_of_output_dir+"/output/ai2rl/"+str(named_config_env)+"/log_avg_lba_arrays.csv"  
    # avg_lba_fileh = create_file(avg_lba_filename) 

    wdGibbsSamp = False 
    if not wdGibbsSamp:
        threshold_stop_Gibbs_sampling = None

    # memory_limit() # Limits RAM usage 
    try:

        # lists_config_method_names = [["rl.sorting_onions_tuning_gae_lambda3"], ["rl.sorting_onions_tuning_LR1"], ["rl.sorting_onions_tuning_LR2"]]  
        lists_config_method_names = [[]]
        for list_config_method_names in lists_config_method_names:  
            named_configs_in = [named_config_env] + list_config_method_names 

            if named_configs_in[0] == patrol_named_config_env: 
                # perimeter patrol
                run_trials_ai2rl(num_trials=num_trials,rootdir=patrol_rootdir_noisy_input,\
                        demonstration_policy_path=patrol_demonstration_policy_path,named_configs_in=named_configs_in,\
                        wdGibbsSamp=wdGibbsSamp, threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling,\
                        total_timesteps_per_session=patrol_total_timesteps_per_session, avg_lba_filename=avg_lba_filename,\
                        hard_lmt_sessioncount=patrol_hard_lmt_sessioncount) 

            if named_configs_in[0] == dmc_named_config_env: 
                # mountain car
                run_trials_ai2rl(num_trials=num_trials,rootdir=dmc_rootdir_noisefree_input,\
                        demonstration_policy_path=dmc_demonstration_policy_path,named_configs_in=named_configs_in,\
                        wdGibbsSamp=wdGibbsSamp, threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling,\
                        total_timesteps_per_session=dmc_total_timesteps_per_session, avg_lba_filename=avg_lba_filename,\
                        hard_lmt_sessioncount=dmc_hard_lmt_sessioncount) 

            if named_configs_in[0] == awn_named_config_env: 
                # ant
                run_trials_ai2rl(num_trials=num_trials,rootdir=awn_rootdir_noisefree_input,\
                        demonstration_policy_path=awn_demonstration_policy_path,named_configs_in=named_configs_in,\
                        wdGibbsSamp=wdGibbsSamp, threshold_stop_Gibbs_sampling=threshold_stop_Gibbs_sampling,\
                        total_timesteps_per_session=awn_total_timesteps_per_session, avg_lba_filename=avg_lba_filename,\
                        hard_lmt_sessioncount=awn_hard_lmt_sessioncount) 


    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
