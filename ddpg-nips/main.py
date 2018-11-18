import argparse
import time
import datetime
import logging
import gym
import os
import sys
import pickle
import tensorflow as tf
import numpy as np
sys.path = ['../'] + sys.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import baselines.common.tf_util as U
from baselines import logger
from subprocess import check_output
from collections import deque
from baselines.common.distributions import make_pdtype
from baselines import logger, bench
from models import Actor, Critic
from memory import Memory, MultiprocessMemory
from noise import *
from osim.env import ProstheticsEnv
from visualize import Visualizer
from util import get_difficulty
from baseclass import BaseClass
from team import EnsembleTeam
from team import EnsembleTeam
from env import EnvGenerator


class DDPGEnsemble(BaseClass):
    # class property
    DEFAULTS = {
        'alias': 'ensemble-ddpg',
        'seed': 0,
        'nb-epochs': 1000,
        'nb-epoch-cycles': 20,
        'nb-eval-steps': 10000,
        'model-dir': '',
        'num-process': 8,
        'run-name': "DDPG-Ensemble",
        'batch-size': 128,
        'nb-train-steps': 50,
        'nb-rollout-steps': 100,
        'evaluation': False,
        'eval-num': 5,
        'HPC': True
    }

    def __init__(self):
        # parse arguments
        super(DDPGEnsemble, self).__init__()
        # prepare before run
        self.prepare()

    def prepare(self):
        cwd = os.path.join(os.getcwd(), 'log')
        self.run_name += datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S") + '-' + str(self.seed)
        cwd = os.path.join(cwd, self.run_name)
        os.makedirs(cwd, exist_ok=True)
        logger.configure(dir=cwd)
        logger.info('Command Line: ' + str(sys.argv))
        # logger.info(str(_) + " is ignored by main parser")

    def run(self):
        # create envs.
        gym.logger.setLevel(logging.ERROR)
        env_generator = EnvGenerator()
        self.env = env_generator.generate(training=True, num_process=self.num_process)
        self.eval_env = env_generator.generate(training=False, num_process=self.eval_num)
        obs_dim = env_generator.get_obs_dim()

        # initial perturbation
        self.eval_env.reset()
        for _ in range(20):
            obs, _, done, info = self.eval_env.step(np.random.rand(self.eval_num, 19))

        self.visualizer = Visualizer(self.alias, self.run_name)
        self.ensemble_team = EnsembleTeam(action_shape=self.env.action_space.shape, observation_shape=(obs_dim,))

        start_time = time.time()
        self.train()
        self.close_env()
        logger.info('total runtime: {}s'.format(time.time() - start_time))


    def train(self):
        self.nb_rollout_steps //= self.num_process
        self.nb_epoch_cycles *= self.num_process
        # nb_train_steps = nb_train_steps * num_process
        saver = tf.train.Saver()
        best_saver = tf.train.Saver()

        step, episode = 0, 0
        best_eval = -1000.
        start_time = time.time()
        episode_reward = np.zeros((self.num_process,), dtype=np.float64)
        episode_step = np.zeros((self.num_process,), dtype=np.int32)
        self.visualizer.initialize('train-reward', 'red')
        if self.eval_env is not None:
            self.visualizer.initialize('eval-reward', 'cyan')
            self.visualizer.initialize('distance*1000', 'green')

        with U.single_threaded_session() as sess:
            print('model_dir', self.model_dir)
            if not self.model_dir:
                self.ensemble_team.initialize(sess)
            else:
                new_saver = tf.train.import_meta_graph(self.model_dir + '.meta')
                new_saver.restore(sess, self.model_dir)
                self.ensemble_team.sess = sess

            self.ensemble_team.add_to_collection("eval")
            sess.graph.finalize()
            self.ensemble_team.reset()
            obs = self.env.reset()
            for epoch in range(self.nb_epochs):
                epoch_reward, epoch_steps, epoch_episode = [], [], 0
                epoch_aloss, epoch_closs, epoch_dis = [], [], []
                epoch_start_time = time.time()
                for cycle in range(self.nb_epoch_cycles):
                    if self.ensemble_team.param_noise:
                        self.ensemble_team.param_noise.decay(steps=step * self.num_process)
                    if self.ensemble_team.action_noise:
                        for noise in self.ensemble_team.action_noise:
                            noise.decay(steps=step * self.num_process)

                    # Rollout
                    for t_rollout in range(self.nb_rollout_steps):
                        # if self.ensemble_team.step_noise and self.ensemble_team.param_noise:
                        #     self.ensemble_team.sess.run(self.ensemble_team.perturb_policy_ops, feed_dict={
                        #         self.ensemble_team.param_noise_stddev: self.ensemble_team.param_noise.current_stddev})
                        action = self.ensemble_team.pi(obs, apply_noise=True, compute_Q=True)

                        new_obs, r, done, info = self.env.step(action)
                        obs_before_reset = [t['obs'] for t in info]
                        self.ensemble_team.store_transition(obs, action, r, obs_before_reset, done)
                        obs = new_obs
                        ###
                        episode_reward += r
                        episode_step += 1
                        step += 1
                        for s in range(self.num_process):
                            if done[s]:
                                epoch_reward.append(episode_reward[s])
                                epoch_steps.append(episode_step[s])
                                episode_step[s] = 0
                                episode_reward[s] = 0
                                episode += 1
                                epoch_episode += 1
                    # Train
                    if self.ensemble_team.get_memory_current_entries() >= self.batch_size:
                        for t_train in range(self.nb_train_steps):
                            c1, a1 = self.ensemble_team.train()
                            self.ensemble_team.update_target_net()
                            epoch_closs.append(c1)
                            epoch_aloss.append(a1)
                        self.ensemble_team.reset()
                # Evaluate
                timer = time.time()
                if self.eval_env is not None:
                    eval_rewards = []
                    eval_forward = []
                    for eval_time in range(self.eval_num):
                        eval_obs = self.eval_env.reset()
                        eval_episode_rewards = np.zeros(self.eval_num)
                        eval_forward = [0 for _ in range(self.eval_num)]
                        dones = [False for _ in range(self.eval_num)]
                        for t_rollout in range(self.nb_eval_steps):
                            mask = 1 - np.array(dones).astype(np.float32)
                            action = self.ensemble_team.pi(eval_obs, apply_noise=False, compute_Q=True)
                            eval_obs, r, current_dones, infos = self.eval_env.step(action)
                            dones = [a or b for a, b in zip(dones, current_dones)]
                            eval_episode_rewards += r * mask
                            for index, (info, mask_val) in enumerate(zip(infos, mask)):
                                if mask_val > 0:
                                    eval_forward[index] = info['pos_x']
                            if all(dones):
                                break
                        eval_rewards.extend(eval_episode_rewards)
                        eval_forward.extend(eval_forward)
                    logger.info('evaluation time:', time.time() - timer)
                    logger.info('Forward distance: ' + str(eval_forward))
                    logger.info('Eval results: ' + str(eval_rewards))
                # Log after an epoch
                is_best = False
                stats = self.ensemble_team.get_stats()
                stats['total/steps'] = step * self.num_process
                stats['total/episode'] = episode
                stats['total/epochs'] = epoch + 1
                stats['total/time'] = time.time() - start_time
                stats['epoch/reward_sum'] = np.mean(epoch_reward)
                stats['epoch/steps'] = np.mean(epoch_steps)
                stats['epoch/episode'] = epoch_episode
                stats['epoch/time'] = time.time() - epoch_start_time
                stats['train/actor_loss'] = np.mean(epoch_aloss)
                stats['train/critic_loss'] = np.mean(epoch_closs)
                # stats['train/param_noise_distance'] = np.mean(epoch_dis)
                if self.eval_env is not None:
                    eval_ave_reward = np.mean(eval_rewards)
                    stats['eval/reward_sum'] = eval_ave_reward
                    if eval_ave_reward > best_eval:
                        best_eval = eval_ave_reward
                        is_best = True
                    stats['eval/mean_forward'] = np.mean(eval_forward)
                for key in sorted(stats.keys()):
                    logger.record_tabular(key, stats[key])
                logger.dump_tabular()
                logger.info('')

                # Visualize
                self.visualizer.paint('train-reward', stats['total/steps'] / 1e6, stats['epoch/reward_sum'])
                if self.eval_env is not None:
                    self.visualizer.paint('eval-reward', stats['total/steps'] / 1e6, stats['eval/reward_sum'])
                    self.visualizer.paint('distance*1000', stats['total/steps'] / 1e6,
                                     stats['eval/mean_forward'] * 1000)

                # Save model
                if saver is not None:
                    saver.save(sess, os.path.join(logger.get_dir(), 'model'), global_step=epoch + 1)
                    if is_best:
                        best_saver.save(sess, os.path.join(logger.get_dir(), 'model-best'),
                                        global_step=epoch + 1)
                if self.HPC and epoch % 5 == 0:
                    try:
                        logger.info(check_output("python copyexist.py", shell=True))
                    except:
                        pass

        def close_env(self):
            self.env.close()
            self.eval_env.close()


if __name__ == '__main__':
    ddpg_ensemble = DDPGEnsemble()
    ddpg_ensemble.run()

