for i in {1..2}
do
    python -m imitation.scripts.train_adversarial airl with sorting_onions demonstrations.rollout_path=/home/katy/imitation/output/rollouts_from_policylist_and_save/imitationNM_SortingOnions-v0/20220820_161939_03c3c0/rollout.pkl
    echo ""
done

echo "Message Body" | mail -s "AIRL experiment finished" sarora@udel.edu
