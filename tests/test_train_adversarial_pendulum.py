import os
from imitation.scripts import train_adversarial
import numpy as np
import time 
import argparse
from test_train_adversarial import test_train_adversarial

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument("noise_insertion_raw_data") 
    parser.add_argument("wdGibbsSamp") 
    parser.add_argument("num_iters_Gibbs_sampling") 
    
    args = parser.parse_args() 

    named_configs_in = ['pendulum','rl.sac', 'train.sac'] 
    demonstration_policy_path = None 
    algorithm_kwargs_tr_adv = dict(
        # Number of discriminator updates after each round of generator updates n_disc_updates_per_round
        n_disc_updates_per_round=4,
        # Equivalent to no replay buffer if batch size is the same gen_replay_buffer_capacity
        # gen_replay_buffer_capacity=16384,
        demo_batch_size=1024, # <= size of rollout
        allow_variable_horizon=True
    )
    rl_batch_size = 2048 # as per hyper param tuning for int(2e4) steps with return_mean as metric
    eval_n_timesteps = algorithm_kwargs_tr_adv['demo_batch_size']
    max_time_steps = eval_n_timesteps + 1
    total_timesteps_per_session = int(1e4)
    rollout_path = './quickstart/pendulum/rl/rollouts/final.pkl'
    env_make_kwargs = {
    }
    if args.noise_insertion_raw_data == 'true':
        noise_insertion_raw_data = True
    else:
        noise_insertion_raw_data = False
    if args.wdGibbsSamp == 'true':
        wdGibbsSamp = True
    else:
        wdGibbsSamp = False
    threshold_stop_Gibbs_sampling = 0.005
    num_iters_Gibbs_sampling = int(args.num_iters_Gibbs_sampling)
    A_B_values_path = '/home/katy/imitation/Results_rAIRL/A_B_values/pendulum/'

    _ = test_train_adversarial(
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
        env_make_kwargs,
        A_B_values_path
    )