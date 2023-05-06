# for a policy hard coded in method 
# python -m imitation.scripts.eval_policy rollouts_from_policylist_and_save

# Perimeter patrol
# env_nm_config=perimeter_patrol  
# policypath="/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"

# Discretized Mountain Car 
# env_nm_config=discretized_state_mountain_car
# policypath="/home/katy/imitation/output/train_rl/imitationNM_DiscretizedStateMountainCar-v0/set3_policy/20230217_160919_a9fee2/policies/final"

# Ant & Ant wd Noise
env_nm_config=ant_wd_noise
policypath="/home/katy/imitation//output/train_rl/Ant-v2/20230505_153134_1ab855/policies/final"

# noiseinsertion=True
noiseinsertion=False

for i in {1..100}
do
    # PPO learned policy 
    python -m imitation.scripts.eval_policy eval_policy with $env_nm_config policy_path=$policypath noise_insertion=$noiseinsertion

    # Hardcoded policy perimeter patrol
    # python -m imitation.scripts.eval_policy eval_policy with perimeter_patrol noise_insertion=$noiseinsertion

    echo ""
done

echo "Message Body" | mail -s "rollouts saved" sarora@udel.edu
