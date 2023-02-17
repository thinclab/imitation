# for a policy hard coded in method 
# python -m imitation.scripts.eval_policy rollouts_from_policylist_and_save

# Perimeter patrol 
# policypath="/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"

# Discretized Mountain Car 
policypath="/home/saurabh/Desktop/Phd_research_work/imitation/output/train_rl/imitationNM_DiscretizedStateMountainCar-v0/set2_policy/20230217_141758_a70bb2/policies/final"

noiseinsertion=False

for i in {1..50}
do
    # PPO learned policy 
    # python -m imitation.scripts.eval_policy eval_policy with perimeter_patrol policy_path=$policypath noise_insertion=$noiseinsertion
    python -m imitation.scripts.eval_policy eval_policy with discretized_state_mountain_car policy_path=$policypath noise_insertion=$noiseinsertion

    # Hardcoded policy perimeter patrol
    # python -m imitation.scripts.eval_policy eval_policy with perimeter_patrol noise_insertion=$noiseinsertion

    echo ""
done

echo "Message Body" | mail -s "rollouts saved" sarora@udel.edu
