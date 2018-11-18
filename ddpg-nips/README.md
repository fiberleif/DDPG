## Code for NIPS 2019 challenge!

**Currently there may be many bugs in the code. If you find one, please report it to me.**

### Usage

#### Installation

1. install Anaconda from https://www.anaconda.com/download/

2. create environment following https://github.com/stanfordnmbl/osim-rl

3. install openai baselines following https://github.com/openai/baselines

4. install visdom following https://github.com/facebookresearch/visdom

#### Train

1. modify YOUR_ALIAS in line 24/main.py and line 5/draw_summary.py

2. run `python main.py --run-name sth`, with options `--seed 10`. to visualize the eval process, add options `--evaluation`

3. watch the training curve (including train reward and eval reward) on http://23.96.14.21:8888/ , switch to environment YOUR_ALIAS

4. to summarize the training, run `python draw_summary.py`. it will automatically merge those log with save run_name and summarize the curve with mean and std, and show it with name 'summarize'.

#### Implementation

If you want to implement your own policy, please inherit `policies/basepolicy.py/BasePolicy` and implement the required methods. You can set all parameters used in your policy in DEFAULTS property. Those parameters will be automatically parsed from command line or default values.

#### Test

Not Implemented yet.

### Support

Currently DDPG and PPO only.

### TODO list

1. add file for test and submit

2. add tricks to PPO

3. add support for other algorithms
