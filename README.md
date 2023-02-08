This is forked and modified from https://github.com/HumanCompatibleAI/imitation in order to implement online AIRL under noise input. 

# Installation 

Set up conda environment
conda env create -f env_HumanCompatibleAIpy3.8.yml

The code is designed to work with a modifed version of stable baselines, https://github.com/thinclab/stable-baselines3 . It is customized to accommodate discrete state discrete action gym environments. Install that repository:  
git clone https://github.com/thinclab/stable-baselines3
cd stable-baselines3
pip install -e .


git clone http://github.com/thinclab/imitation
cd imitation
pip install -e .


# Running Online AIRL 

In scripts/config directory, set up configs you want to use for training expert in config file for imitation.scripts.train_rl (PPO hyperparameters, training time, etc)

Train expert using 'python -m imitation.scripts.train_rl with <env name config>'
example: python -m imitation.scripts.train_rl with perimeter_patrol

In scripts/config directory, set up configs (size etc.) you want to use for simulating trajectories (called rolllouts) imitation.scripts.eval_policy from expert's policy. Please make sure you have set up a common formatted path where all rollouts should be saved. 

Use path of saved policy and create_rollouts_from_policy.sh to create rollouts for different sessions of IRL 

Set input arguments in tests/test_call_ai2rl.py like path of directory where all rollouts are saved, number of times you want to run online IRL, env name, etc. 

Execute python tests/test_call_ai2rl.py 

# Output

The learned behavior accuracy is measure of match between action choices of expert/ true policy and learner's policy. For discrete state discrete action domain, this metric is computed as (number of matching pairs of state and action)*100/(number of states). For continuous state domain, the state space is partitioned and metric is computed as (number of matching pairs of sampled state and chosen action)*100/(size of state space), integrated over each partition and summed over all partitions. The integral is approximated using Monte Carlo. 

Each trial of online AIRL should generate an array of LBA values. Code averges these values  over all trials, resulting in an array of (average, stddev) value per session. This value should go higher with more sessions eventually plateauing. 

# Citations (BibTeX)
```
@misc{arora2022imitation,
  author = {Arora, Saurabh},
  title = {{\tt imitation} Library for Online Inverse Reinforcement Learning Under Noisy Input},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HumanCompatibleAI/imitation}},
}
```

Source repository
```
@misc{wang2020imitation,
  author = {Wang, Steven and Toyer, Sam and Gleave, Adam and Emmons, Scott},
  title = {The {\tt imitation} Library for Imitation Learning and Inverse Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HumanCompatibleAI/imitation}},
}
```

