import os
from imitation.scripts import eval_policy
import numpy as np
import time 
import git 
import os 

repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_home = repo.working_tree_dir
parent_of_output_dir = str(git_home)


def test_eval_policy(policypath,noiseinsertion,env_name_config, save_videos, render=False):
    config_updates = {
        "policy_path": policypath,
        "noise_insertion": noiseinsertion,
        "videos": save_videos
    }
    if env_name_config:
        if render:
            run = eval_policy.eval_policy_ex.run(
                command_name='eval_policy',
                named_configs=[env_name_config,'render'],
                config_updates=config_updates,
            )
        else:
            run = eval_policy.eval_policy_ex.run(
                command_name='eval_policy',
                named_configs=[env_name_config],
                config_updates=config_updates,
            )
    else:
        run = eval_policy.eval_policy_ex.run(
            command_name='eval_policy',
            config_updates=config_updates,
        )
    assert run.status == "COMPLETED"
    print(run.result)


if __name__ == '__main__':

    save_videos = False

    policypath_ant = parent_of_output_dir+"/output/train_rl/Ant-v2/20230505_153134_1ab855/policies/final"
    env_name_config_ant = 'ant' 
    env_name_antwdnoise = 'ant_wd_noise'

    policypath_reacher = parent_of_output_dir+'/output/train_rl/Reacher-v2/20230505_183718_c7b35f/policies/final'
    env_name_config_reacher = 'reacher' 

    env_name_config_half_cheetah = 'half_cheetah' 
    # expert policies
    policypath_half_cheetah = parent_of_output_dir+'/output/train_rl/HalfCheetah-v3/20230511_191304_8315b7/policies/000001000000'
    policypath_half_cheetah2_0pnt01noise = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/other_noise_levels/20230512_121542_f7216c/policies/000001000000'
    policypath_half_cheetah2_0pnt05noise = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/other_noise_levels/20230512_133554_e533db/policies/000003000000'
    policypath_half_cheetah2_0pnt075noise = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/other_noise_levels/20230512_140918_7c19d7/policies/00003000000'
    policypath_half_cheetah2_0pnt09noise = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/other_noise_levels/20230512_151327_a43c3a/policies/000000050000'
    policypath_half_cheetah2_0pnt095noise_1pnt25forwd = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/other_noise_levels/20230512_162644_9419d1/policies/000000990000'
    policypath_half_cheetah2_0pnt09noise_mrTrn = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/other_noise_levels/20230512_180946_95009d/policies/000000530000'
    # chosen expert
    policypath_half_cheetah2_0pnt095noise = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdWeights-v0/0pnt095noise/20230512_155826_16533b/policies/000000500000'
    # learned policies 
    policypath_half_cheetah2_learner_wo_noise_session10_75000trainsteps = parent_of_output_dir+'/output/airl/imitationNM_HalfCheetahEnvMdfdWeights-v0/noise_free/demo_size_8192_trainsteps_75000/20230514_173931_51f5cc/checkpoints/final/gen_policy'
    policypath_half_cheetah2_learner_wo_noise_session51_75000trainsteps = parent_of_output_dir+'/output/airl/imitationNM_HalfCheetahEnvMdfdWeights-v0/noise_free/demo_size_8192_trainsteps_75000/20230514_192624_c6dd70/checkpoints/final/gen_policy'
    policypath_half_cheetah2_learner_wo_noise_session75_75000trainsteps = parent_of_output_dir+'/output/airl/imitationNM_HalfCheetahEnvMdfdWeights-v0/noise_free/demo_size_8192_trainsteps_75000/20230514_202218_cd3685/checkpoints/final/gen_policy'
    policypath_half_cheetah2_learner_wo_noise_session85_75000trainsteps = parent_of_output_dir+'/output/airl/imitationNM_HalfCheetahEnvMdfdWeights-v0/noise_free/demo_size_8192_trainsteps_75000/20230514_204440_64487a/checkpoints/final/gen_policy'
    policypath_half_cheetah2_learner_wo_noise_session100_75000trainsteps = parent_of_output_dir+'/output/airl/imitationNM_HalfCheetahEnvMdfdWeights-v0/noise_free/demo_size_8192_trainsteps_75000/20230514_211815_ab9312/checkpoints/final/gen_policy'
    policypath_half_cheetah2_learner_wo_noise_session50_75000trainsteps = parent_of_output_dir+'/output/airl/imitationNM_HalfCheetahEnvMdfdWeights-v0/noise_free/demo_size_256_trainsteps_75000/20230524_090443_ad329d/checkpoints/final/gen_policy'

    env_name_config_half_cheetah2 = 'half_cheetah_mdfd_weights'
    policypath_half_cheetah3 = parent_of_output_dir+'/output/train_rl/imitationNM_HalfCheetahEnvMdfdReward-v0/20230512_165157_29de4a/policies/000000030000'
    env_name_config_half_cheetah3 = 'half_cheetah_mdfd_reward'

    env_name_config_hp_sac = 'hopper_sac' 
    # expert policies
    policypath_hp_customsac = parent_of_output_dir+'/output/train_rl/Hopper-v3/20230608_120236_60b2f7/policies/000001000000'
    # above failed bcz some use_sde argument repeated in policy object creation 
    policypath_hp_defaultsac = parent_of_output_dir+'/output/train_rl/Hopper-v3/20230608_194557_5321f1/policies/000001000000'
    env_name_config_hp_ppo = 'hopper_ppo' 
    policypath_hp_ppo = parent_of_output_dir+'/output/train_rl/Hopper-v3/20230609_092251_d8bbea/policies/000001000000'
    
    noiseinsertion = True 
    render = True

    for i in range(1):
        test_eval_policy(policypath_hp_ppo, noiseinsertion, \
                     env_name_config_hp_ppo, save_videos, render) 