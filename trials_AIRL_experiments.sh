
for i in {1..2}
do
    # args noise_insertion_raw_data wdGibbsSamp num_iters_Gibbs_sampling

    # airl without noisy input 
    python ./tests/test_train_adversarial.py false false 25 > "./Results_rAIRL/wo_noise/output$i.txt" &
    # airl with noisy input 
    # python ./tests/test_train_adversarial.py true false 25 > "./Results_rAIRL/wd_noise_wd_gibbs/0.01noise/output_wdnswdgb_$i.txt" &
    # robust airl with noisy input
    # python ./tests/test_train_adversarial.py true true 25 > "./Results_rAIRL/wd_noise_wo_gibbs/0.01noise/output_wdnswogb_$i.txt" &
    echo ""
done

echo "Message Body" | mail -s "AIRL experiment finished" sarora@udel.edu
