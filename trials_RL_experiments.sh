# format
# python -m imitation.scripts.train_rl with sorting_onions common.fast train.fast rl.fast fast common.log_dir=quickstart/rl/

for i in {1..10}
do
    python -m imitation.scripts.train_rl with perimeter_patrol
    echo ""
done

echo "Message Body" | mail -s "RL experiment finished" sarora@udel.edu
