import os
from imitation.scripts import train_adversarial
import numpy as np
import time 


def test_train_adversarial(
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
):
    config_updates = {
        "agent_path": None,
        "demonstrations": dict(rollout_path=rollout_path),
        "demonstration_policy_path": demonstration_policy_path,
        "wdGibbsSamp": wdGibbsSamp,
        "threshold_stop_Gibbs_sampling": threshold_stop_Gibbs_sampling,
        "total_timesteps": total_timesteps_per_session,
        "num_iters_Gibbs_sampling": num_iters_Gibbs_sampling,
        "noise_insertion_raw_data": noise_insertion_raw_data,
        "eval_n_timesteps": eval_n_timesteps,
        "max_time_steps": max_time_steps,
        "algorithm_kwargs": algorithm_kwargs_tr_adv,
        "rl_batch_size": rl_batch_size,
        "env_make_kwargs": env_make_kwargs
    }
    
    run = train_adversarial.train_adversarial_ex.run(
        command_name='airl',
        named_configs=named_configs_in,
        config_updates=config_updates,
    )

    assert run.status == "COMPLETED"
    print(run.result)

if __name__ == '__main__':

    named_configs_in = ['soContSpaces']
    demonstration_policy_path = None
    algorithm_kwargs_tr_adv = dict(
        # Number of discriminator updates after each round of generator updates n_disc_updates_per_round
        n_disc_updates_per_round=4,
        # Equivalent to no replay buffer if batch size is the same gen_replay_buffer_capacity
        # gen_replay_buffer_capacity=16384,
        demo_batch_size=32, # <= size of rollout
        allow_variable_horizon=True
    )
    rl_batch_size = 32
    eval_n_timesteps = algorithm_kwargs_tr_adv['demo_batch_size']
    max_time_steps = eval_n_timesteps + 1
    total_timesteps_per_session = int(6e4)
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
    noise_insertion_raw_data = True
    wdGibbsSamp = True
    threshold_stop_Gibbs_sampling = 0.02
    num_iters_Gibbs_sampling = 20

    test_train_adversarial(
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