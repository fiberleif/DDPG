from copy import copy
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines import logger
import baselines.common.tf_util as U
from util import reduce_std, Adam, RunningMeanStd
from baseclass import BaseClass
from agent import DDPGAgent
from baselines.common.misc_util import set_global_seeds
from memory import Memory, MultiprocessMemory
from noise import *

class EnsembleTeam(BaseClass):
    DEFAULTS = {
        'seed': 0,
        'memory-size': 1e6,
        'batch-size': 128,
        'noise-type': 'ou_0.1',  # choices are adaptive-param_xx, ou_xx, normal_xx, none
        'ensemble-num': 5,
        'num-process': 8,
        'final-steps': 3e6,
        'final-action-noise': 0.003,
        'pn-init': 0.00016,
        'ou-dec': 1e-7,
        'reward-scale': 1.0,
        'normalize-returns': False,
        'normalize-observations': True,
    }

    def __init__(self, observation_shape, action_shape):
        # parse arguments
        super(EnsembleTeam, self).__init__()
        self.memory_size = int(self.memory_size)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.action_range = (-1., 1.)
        self.stats_sample = None
        # seed everything to make things reproducible.
        tf.reset_default_graph()
        set_global_seeds(self.seed)
        # configure components
        self._configure_component()
        # init graph

    def _configure_component(self):
        # configure multiple actor-critic components
        self.agents = []
        self.obs0 = []
        self.actions = []
        self.actor_tf = []
        self.critic_tf = []
        for i in range(self.ensemble_num):
            agent = DDPGAgent(i, self.observation_shape, self.action_shape)
            self.agents.append(agent)
            self.obs0.append(agent.obs0)
            self.actions.append(agent.actions)
            self.actor_tf.append(agent.actor_tf)
            self.critic_tf.append(agent.critic_tf)
        # configure memory component
        self.memory = Memory(limit=self.memory_size, action_shape=self.action_shape,
                             observation_shape=self.observation_shape)
        # configure noise component
        self._parse_noise()

    def _parse_noise(self):
        self.action_noise = None
        self.param_noise = None
        nb_actions = self.action_shape[-1]
        for current_noise_type in self.noise_type.split('*'):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=self.pn_init, desired_action_stddev=float(stddev),
                        final_steps=self.final_steps, final_desired=self.final_action_noise)
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                        sigma=float(stddev) * np.ones(nb_actions), dec=self.ou_dec, final_steps=self.final_steps,
                        final_sigma=self.final_action_noise) for _ in range(self.num_process)]
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))


    def initialize(self, sess):
        """
        init parameter of all networks.
        :return:
        """
        # self.sess = tf.Session()
        # self.sess = U.single_threaded_session()
        # self.sess = tf.InteractiveSession()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        # init target network to the same
        for agent in self.agents:
            agent.initialize(self.sess)

    def reset(self):
        """
        reset noise during rollout.
        :return:
        """
        if self.action_noise is not None:
            for noise in self.action_noise:
                noise.reset()

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        for i in range(len(obs0)):
            self.memory.append(obs0[i], action[i], reward[i] * self.reward_scale, obs1[i], terminal1[i])
            if self.normalize_observations:
                self.agents[0].obs_rms.update(np.array([obs0[i]]))

    def add_to_collection(self, collection_name):
        for agent in self.agents:
            tf.add_to_collection(collection_name, agent.obs0)
            tf.add_to_collection(collection_name, agent.actions)
            tf.add_to_collection(collection_name, agent.actor_tf)
            tf.add_to_collection(collection_name, agent.critic_tf)

    def pi(self, eval_obs, apply_noise=True, compute_Q=True):
        actions = [agent.pi(eval_obs, compute_Q=False)[0] for agent in self.agents]
        action_with_max_q_value = None
        max_q_value = - float("inf")
        for action in actions:
            sum = 0
            for agent in self.agents:
                feed_dict = {agent.obs0: eval_obs, agent.actions: action}
                sum += np.sum(self.sess.run(agent.critic_tf, feed_dict=feed_dict))
            if sum >= max_q_value:
                max_q_value = sum
                action_with_max_q_value = action
        if apply_noise:
            noise = [noise() for noise in self.action_noise]
            noise = np.array(noise)
            action_with_max_q_value += noise
        action_clip = np.clip(action_with_max_q_value, self.action_range[0], self.action_range[1])
        return action_clip

    def train(self):
        actor_loss_sum, critic_loss_sum = 0., 0.
        for agent in self.agents:
            # get a batch.
            batch = self.memory.sample(batch_size=self.batch_size)
            avg_target_Q = self.average_target_q(batch)
            critic_loss, actor_loss = agent.train(batch, avg_target_Q)
            actor_loss_sum += actor_loss
            critic_loss_sum += critic_loss
        return critic_loss_sum, actor_loss_sum

    def average_target_q(self, batch):
        sum_target_q = np.zeros((self.batch_size, 1))
        for idx, agent in enumerate(self.agents):
            if self.normalize_returns and self.enable_popart:
                old_mean, old_std, target_Q = self.sess.run([agent.ret_rms.mean, agent.ret_rms.std, agent.target_Q],
                                                            feed_dict={
                                                                agent.obs1: batch['obs1'],
                                                                agent.rewards: batch['rewards'],
                                                                agent.terminals1: batch['terminals1'].astype('float32'),
                                                            })
                if idx == 0:
                    agent.ret_rms.update(target_Q.flatten())
                agent.sess.run(self.renormalize_Q_outputs_op, feed_dict={
                    agent.old_std: np.array([old_std]),
                    agent.old_mean: np.array([old_mean]),
                })

            else:
                target_Q = self.sess.run(agent.target_Q, feed_dict={
                    agent.obs1: batch['obs1'],
                    agent.rewards: batch['rewards'],
                    agent.terminals1: batch['terminals1'].astype('float32'),
                })
            assert sum_target_q.shape == target_Q.shape
            sum_target_q += target_Q
        avg_target_q = sum_target_q / self.ensemble_num
        return avg_target_q

    def update_target_net(self):
        for agent in self.agents:
            agent.update_target_net()

    def get_stats(self):
        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        from collections import Counter
        sum_stats = Counter({})
        for agent in self.agents:
            stats = agent.get_stats(self.stats_sample)
            assert type(stats) == dict
            sum_stats += Counter(stats)
        sum_stats = dict(sum_stats)
        return sum_stats

    def get_memory_current_entries(self):
        return self.memory.nb_entries



