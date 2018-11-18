import argparse
import os
import tensorflow as tf
import numpy as np
import sys
sys.path = ['../'] + sys.path
import baselines.common.tf_util as U
import gym
from osim.env import ProstheticsEnv
from osim.http.client import Client
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from util import get_difficulty
from observation import ObsProcessWrapper, RewardReshapeWrapper
remote_base = ["http://grader.crowdai.org:1729", "http://grader.crowdai.org:1730"]
crowdai_token = "5a42310a409d7c3ee3bdadb3347a869d"

class OSmodel:
    ## simulate local env
    def __init__(self):
        self.istep = 0

class RemoteProstheticsEnv(gym.Env):
    def __init__(self, base, token, round):
        self.base = base
        self.token = token
        self.client = None
        ## simulate local env
        self.osim_model = OSmodel()
        self.time_limit = 300 if round == 1 else 1000

    def reset(self, project=True):
        if self.client == None:
            self.client = Client(self.base)
            obs = self.client.env_create(self.token, env_id='ProstheticsEnv')
            self.osim_model.istep = 0
            return obs
        else:
            ### It is not allowed to call reset() twice in submitting.
            raise NotImplementedError
    def step(self, action, project=True):
        self.osim_model.istep += 1
        [obs, reward, done, info] = self.client.env_step(action.tolist(), render=True)
        if done:
            self.osim_model.istep = 0
            obs = self.client.env_reset()
            if not obs:
                done = True
            else:
                done = False
        return obs, reward, done, info

def run(online, models, round, visualize, old_version):
    # Load Models
    print("Start loading models...")
    cwd = os.getcwd()
    sess_a, ops_a = [], []
    for name in models:
        pwd = os.path.join(cwd, name)
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(pwd+'.meta')
        new_saver.restore(sess, pwd)
        sess_a.append(sess)
        ops_a.append(tf.get_collection('eval'))
        print("Load model {} success.".format(pwd))

    # Init env (TODO: online)
    difficulty = get_difficulty(round)
    if not online:
        env = ProstheticsEnv(visualize=visualize, difficulty=difficulty)
    else:
        env = RemoteProstheticsEnv(remote_base[difficulty], crowdai_token, round)

    env = ObsProcessWrapper(env, add_feature=True, round=round, old_version=old_version)
    total_r = 0.
    print("Prepare to eval.")
    for s in range(1 if online else 2):
        obs = env.reset()
        done = False
        step = 0

        # Start eval
        while not done:
            env.total_step = env.unwrapped.osim_model.istep
            step += 1
            print(env.total_step)
            actions, critics = [], []
            for sess, ops in zip(sess_a, ops_a):
                a = sess.run(ops[2], feed_dict={ops[0]: [obs]})
                actions.append(a[0])
            for a in actions:
                total_v = 0.
                for sess, ops in zip(sess_a, ops_a):
                    total_v += sess.run(ops[3], feed_dict={ops[0]: [obs], ops[1]: [a]})[0]
                critics.append(total_v)
            max_ind = np.argmax(critics)
            best_a = actions[max_ind]
            obs, r, done, info = env.step(best_a)
            if r:
                total_r += r
            else:
                break

    print('Eval End: total reward {}'.format(total_r / 2))
    if online:
        env.unwrapped.client.submit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    boolean_flag(parser, 'online', default=False)
    boolean_flag(parser, 'visualize', default=False)
    boolean_flag(parser, 'old-version', default=False)
    parser.add_argument('--model', nargs='+', type=str)
    parser.add_argument('--round', type=int, default=1, choices=[1, 2])

    args = parser.parse_args()
    run(args.online, args.model, args.round, args.visualize, args.old_version)