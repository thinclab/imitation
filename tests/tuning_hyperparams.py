from test_train_adversarial import test_train_adversarial
import git 
import resource
import os

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)

if __name__ == '__main__':

    named_configs_in = ['soContSpaces']
    demonstration_policy_path = None
    total_timesteps_per_session = int(2e4)
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
    noise_insertion_raw_data = False
    wdGibbsSamp = False
    threshold_stop_Gibbs_sampling = 0.02
    num_iters_Gibbs_sampling = 2

    runing_results_f = parent_of_output_dir+"/output/ai2rl/so_ContSpaces/hyperparam_tune.csv"
    max_res = 0 

    writer_tuning = open(runing_results_f, "w")
    for dbs in [32,128,256]: # don't have more data than that 
        for rbs in [32,128,512,2048]: 

            algorithm_kwargs_tr_adv = dict(
                # Number of discriminator updates after each round of generator updates n_disc_updates_per_round
                n_disc_updates_per_round=4,
                # Equivalent to no replay buffer if batch size is the same gen_replay_buffer_capacity
                # gen_replay_buffer_capacity=16384,
                demo_batch_size=dbs, # <= size of rollout
                allow_variable_horizon=True
            )
            rl_batch_size = rbs
            eval_n_timesteps = algorithm_kwargs_tr_adv['demo_batch_size']
            max_time_steps = eval_n_timesteps + 1

            result = test_train_adversarial(
                named_configs_in,
                rollout_path,
                demonstration_policy_path,
                wdGibbsSamp,
                threshold_stop_Gibbs_sampling,
                total_timesteps_per_session, 
                num_iters_Gibbs_sampling,
                noise_insertion_raw_data,
                max_time_steps,
                eval_n_timesteps,
                algorithm_kwargs_tr_adv,
                rl_batch_size,
                env_make_kwargs
            )
            
            linestr = ''+str(dbs)+', '+str(rbs)+', '+str(result['imit_stats']["return_mean"])
            writer_tuning.write(linestr+'\n')

            if result['imit_stats']["return_mean"] > max_res:
                max_res = result['imit_stats']["return_mean"]
                writer_tuning.write('max_res updated \n')
        
    writer_tuning.close()