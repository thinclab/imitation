# format
# python -m imitation.scripts.train_rl with discretized_state_mountain_car common.fast train.fast rl.fast fast common.log_dir=quickstart/rl/

for i in {1..10}
do
    python -m imitation.scripts.train_rl with discretized_state_mountain_car

    echo ""
done

echo "Message Body" | mail -s "RL experiment finished" sarora@udel.edu
