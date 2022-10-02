# for a policy hard coded in method 
# python -m imitation.scripts.eval_policy rollouts_from_policylist_and_save

# Perimeter patrol 
policypath="/home/katy/imitation/output/train_rl/imitationNM_PatrolModel-v0/20220923_142937_f57e0c/policies/final"
for i in {1..50}
do
    python -m imitation.scripts.eval_policy eval_policy with perimeter_patrol policy_path=$policypath noise_insertion=True
    echo ""
done

echo "Message Body" | mail -s "rollouts saved" sarora@udel.edu
