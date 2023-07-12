
for i in {1..1}
do
    python ./tests/test_train_adversarial.py > ./Results_rAIRL/wd_noise_wd_Gibbs/0.01noise/output_wdnswdgb_$i.txt &
    echo ""
done

echo "Message Body" | mail -s "AIRL experiment finished" sarora@udel.edu
