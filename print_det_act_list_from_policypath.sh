basepath=/home/katy/imitation/output/train_rl/imitationNM_SortingOnions-v0

python -m imitation.scripts.eval_policy average_lba_for_det_act_list_from_basedir_policypath with rootdir=$basepath

# python -m imitation.scripts.eval_policy lba_for_det_act_list_from_policypath with policy_path=/home/katy/imitation/output/train_rl/imitationNM_SortingOnions-v0/20220819_132642_2be5ab/policies/final
