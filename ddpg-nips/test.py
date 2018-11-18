from osim.env import ProstheticsEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import numpy as np


def make_env(rank, log_dir=None):
    def _thunk():
        env = ProstheticsEnv(visualize=False)
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

envs = [make_env(i, '.') for i in range(4)]
envs = SubprocVecEnv(envs)

obs = envs.reset()
for i in range(2000):
    obs, reward, done, info = env.step([env.action_space.sample() for i in range(4)])
