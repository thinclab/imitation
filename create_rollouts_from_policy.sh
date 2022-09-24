# for a policy hard coded in method 
# python -m imitation.scripts.eval_policy rollouts_from_policylist_and_save

# for a policy learned by rl 
# policypath="/home/katy/imitation/output/train_rl/imitationNM_SortingOnions-v0/default_params/20220819_132642_2be5ab/policies/final"
# python -m imitation.scripts.eval_policy eval_policy with policy_path=$policypath 

policypath="/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"
python -m imitation.scripts.eval_policy eval_policy with perimeter_patrol policy_path=$policypath 
