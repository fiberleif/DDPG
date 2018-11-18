import gym
from osim.env import ProstheticsEnv
from baselines import logger, bench
import os
from env_wrapper import ObsProcessWrapper, RewardReshapeWrapper, FinalObsWrapper, SkipframeWrapper
from util import get_difficulty
from baseclass import BaseClass
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_env(rank, skipcnt, log_dir=None, reward_process=False, reward_bonus=1.0, obs_process=True, round=2,
             final_process=True, skip_frame_process=True, **kwargs):
    def _thunk():
        env = ProstheticsEnv(visualize=False, difficulty=get_difficulty(round))
        if obs_process:
            env = ObsProcessWrapper(env, round=round, **kwargs)
        if reward_process:
            env = RewardReshapeWrapper(env, reward_bonus)
        # ProstheticsEnv have no seed
        #env.seed(seed + 100000 * rank)
        if final_process:
            env = FinalObsWrapper(env)
        if skip_frame_process:
            env = SkipframeWrapper(env, skipcnt)
        # env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

class EnvGenerator(BaseClass):
    DEFAULTS = {
        'round': 2,
        'skip-frame': 2,
        'reward-process': True,
        'reward-bonus': -8.4375,
        'obs-process': True,
        'add-feature': True,
        'old-version': False,
        'y-axis': False,
    }

    def __init__(self):
        # parse arguments
        super(EnvGenerator, self).__init__()

    def generate(self, training=True, num_process=8):
        if training:
            env = [make_env(i, self.skip_frame, log_dir=logger.get_dir(), reward_process=self.reward_process,
                                 reward_bonus=self.reward_bonus, obs_process=self.obs_process, add_feature=self.add_feature,
                                 round=self.round, y_axis=self.y_axis, old_version=self.old_version)
                        for i in range(num_process)]
        else:
            env = [make_env(i, 1, log_dir=logger.get_dir(), reward_process=False,
                                 reward_bonus=self.reward_bonus, obs_process=self.obs_process, add_feature=self.add_feature,
                                 round=self.round, y_axis=self.y_axis, old_version=self.old_version, final_process=False,
                                 skip_frame_process=False)
                        for i in range(num_process)]
        env = SubprocVecEnv(env)
        return env

    def get_obs_dim(self):
        env = ProstheticsEnv(visualize=False, difficulty=get_difficulty(self.round))
        if self.obs_process:
            env = ObsProcessWrapper(env, add_feature=self.add_feature, round=self.round, y_axis=self.y_axis,
                                    old_version=self.old_version)
        return len(env.reset())