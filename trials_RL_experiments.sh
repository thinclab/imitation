
for i in {1..9}
do
    # python -m imitation.scripts.train_rl with sorting_onions common.fast train.fast rl.fast fast common.log_dir=quickstart/rl/
    python -m imitation.scripts.train_rl with sorting_onions # common.log_dir=quickstart/rl/
    echo ""
done

echo "Message Body" | mail -s "RL experiment finished" sarora@udel.edu
