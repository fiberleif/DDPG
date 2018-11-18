from copy import copy
from functools import reduce
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from baselines import logger
import baselines.common.tf_util as U
from util import normalize, denormalize, reduce_std, Adam, RunningMeanStd
from baseclass import BaseClass
from models import Actor, Critic

class DDPGAgent(BaseClass):
    activation_with_name = {'relu': tf.nn.relu, 'selu': tf.nn.selu, 'elu': tf.nn.elu}
    DEFAULTS = {
            'layer-norm': True,
            'normalize-returns': False,
            'normalize-observations': True,
            'critic-l2-reg': 1e-2,
            'actor-lr': 1e-4,
            'critic-lr': 1e-3,
            'clip-norm': None,
            'gamma': 0.99,
            'tau': 0.001,
            'trot': 1,
            'reward-scale': 1.0,
            'activation': 'selu',
            'layer-num': 2,
            'layer-width': 64,
        }

    def __init__(self, index, observation_shape, action_shape):
        super(DDPGAgent, self).__init__()
        self.activation = self.activation_with_name[self.activation]
        # other hyper-parameters.
        self.action_range = (-1., 1.)
        self.return_range = (-np.inf, np.inf)
        self.observation_range = (-5., 5.)
        self.enable_popart = False

        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.critic = Critic(name='critic_{}th'.format(index), layer_norm=self.layer_norm, withaction=True,
                             activation=self.activation, layer_num=self.layer_num, layer_width=self.layer_width)
        self.actor = Actor(action_shape[0], name='actor_{}th'.format(index), layer_norm=self.layer_norm,
                           activation=self.activation, layer_num=self.layer_num, layer_width=self.layer_width)
        self._build_graph()

    # Build computaion graph from here. (Inner)
    def _build_graph(self):
        """
        build computation graph of agent itself.
        """
        self._placeholder()
        self._normalize_obs()
        self._normalize_return()
        self._create_target_networks()
        self._bulid_core_part()
        self._set_up()

    def _placeholder(self):
        # inputs.
        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape, name='obs0')
        self.obs1 = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape, name='obs1')
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + self.action_shape, name='actions')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        # self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

    def _normalize_obs(self):
        # normalize observation.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms', reuse=tf.AUTO_REUSE):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
        else:
            self.obs_rms = None
        self.normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms), self.observation_range[0],
                                           self.observation_range[1])
        self.normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms), self.observation_range[0],
                                           self.observation_range[1])

    def _normalize_return(self):
        # normalize return.
        if self.normalize_returns:
            with tf.variable_scope('ret_rms', reuse=tf.AUTO_REUSE):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

    def _create_target_networks(self):
        # create target networks.
        self.target_actor = copy(self.actor)
        self.target_actor.name = 'target_' + self.actor.name
        self.target_critic = copy(self.critic)
        self.target_critic.name = 'target_' + self.critic.name

    def _bulid_core_part(self):
        # build core TF parts that are shared across setup parts.
        self.actor_tf = self.actor(self.normalized_obs0)
        self.normalized_critic_tf = self.critic(self.normalized_obs0, self.actions)
        self.critic_tf = denormalize(tf.clip_by_value(self.normalized_critic_tf, self.return_range[0],
                                                      self.return_range[1]), self.ret_rms)
        self.normalized_critic_with_actor_tf = self.critic(self.normalized_obs0, self.actor_tf, reuse=True)
        self.critic_with_actor_tf = denormalize(tf.clip_by_value(self.normalized_critic_with_actor_tf,
                                                self.return_range[0], self.return_range[1]), self.ret_rms)
        Q_obs1 = denormalize(self.target_critic(self.normalized_obs1, self.target_actor(self.normalized_obs1)),
                             self.ret_rms)
        self.target_Q = self.rewards + (1. - self.terminals1) * self.gamma * Q_obs1

    def _set_up(self):
        # set up parts.
        # if self.param_noise is not None:
        #     self._setup_param_noise(self.normalized_obs0)
        self._setup_actor_optimizer()
        self._setup_critic_optimizer()
        if self.normalize_returns and self.enable_popart:
            self._setup_popart()
        self._setup_stats()
        self._setup_target_network_updates()

    def _setup_actor_optimizer(self):
        logger.info('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.critic_with_actor_tf)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = U.flatgrad(self.actor_loss, self.actor.trainable_vars, clip_norm=self.clip_norm)
        self.actor_optimizer = Adam(var_list=self.actor.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_critic_optimizer(self):
        logger.info('setting up critic optimizer')
        normalized_critic_target_tf = tf.clip_by_value(normalize(self.critic_target, self.ret_rms),
                                                       self.return_range[0], self.return_range[1])
        self.critic_loss = tf.reduce_mean(tf.square(self.normalized_critic_tf - normalized_critic_target_tf))
        if self.critic_l2_reg > 0.:
            critic_reg_vars = [var for var in self.critic.trainable_vars
                               if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                logger.info('  regularizing: {}'.format(var.name))
            logger.info('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.critic_grads = U.flatgrad(self.critic_loss, self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.critic_optimizer = Adam(var_list=self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def _setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        self.old_std = tf.placeholder(tf.float32, shape=[1], name='old_std')
        new_std = self.ret_rms.std
        self.old_mean = tf.placeholder(tf.float32, shape=[1], name='old_mean')
        new_mean = self.ret_rms.mean

        self.renormalize_Q_outputs_op = []
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1
            self.renormalize_Q_outputs_op += [M.assign(M * self.old_std / new_std)]
            self.renormalize_Q_outputs_op += [b.assign((b * self.old_std + self.old_mean - new_mean) / new_std)]

    def _setup_stats(self):
        ops = []
        names = []

        if self.normalize_returns:
            ops += [self.ret_rms.mean, self.ret_rms.std]
            names += ['ret_rms_mean', 'ret_rms_std']

        if self.normalize_observations:
            ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
            names += ['obs_rms_mean', 'obs_rms_std']

        ops += [tf.reduce_mean(self.critic_tf)]
        names += ['reference_Q_mean']
        ops += [reduce_std(self.critic_tf)]
        names += ['reference_Q_std']

        ops += [tf.reduce_mean(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_mean']
        ops += [reduce_std(self.critic_with_actor_tf)]
        names += ['reference_actor_Q_std']

        ops += [tf.reduce_mean(self.actor_tf)]
        names += ['reference_action_mean']
        ops += [reduce_std(self.actor_tf)]
        names += ['reference_action_std']

        # if self.param_noise:
        #     ops += [tf.reduce_mean(self.perturbed_actor_tf)]
        #     names += ['reference_perturbed_action_mean']
        #     ops += [reduce_std(self.perturbed_actor_tf)]
        #     names += ['reference_perturbed_action_std']

        self.stats_ops = ops
        self.stats_names = names

    def _setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = self._get_target_updates(self.actor.vars,
                                                                          self.target_actor.vars)
        critic_init_updates, critic_soft_updates = self._get_target_updates(self.critic.vars,
                                                                            self.target_critic.vars)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def _get_target_updates(self, vars, target_vars):
        logger.info('setting up target updates ...')
        soft_updates = []
        init_updates = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            logger.info('  {} <- {}'.format(target_var.name, var.name))
            init_updates.append(tf.assign(target_var, var))
            soft_updates.append(tf.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
        assert len(init_updates) == len(vars)
        assert len(soft_updates) == len(vars)
        return tf.group(*init_updates), tf.group(*soft_updates)

    # Create user interface from here. (outer)
    def initialize(self, sess):
        self.sess = sess
        self.sess.run(self.target_init_updates)

    def pi(self, obs, compute_Q=True):
        feed_dict = {self.obs0: obs}
        if compute_Q:
            action, q = self.sess.run([self.actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(self.actor_tf, feed_dict=feed_dict)
            q = None
        action = np.clip(action, self.action_range[0], self.action_range[1])
        return action, q

    def train(self, batch, avg_target_Q):
        actor_loss_sum, critic_loss_sum = 0., 0.
        # Trot: sample one batch and use it to implement back-propagation for multiple times with a small learning rate
        for _trot in range(self.trot):
            # Get all gradients and perform a synced update.
            ops = [self.actor_grads, self.actor_loss, self.critic_grads, self.critic_loss]
            actor_grads, actor_loss, critic_grads, critic_loss = self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.critic_target: avg_target_Q,
            })
            self.actor_optimizer.update(actor_grads, stepsize=self.actor_lr / self.trot)
            self.critic_optimizer.update(critic_grads, stepsize=self.critic_lr / self.trot)
            actor_loss_sum += actor_loss
            critic_loss_sum += critic_loss

        return critic_loss_sum, actor_loss_sum

    def update_target_net(self):
        self.sess.run(self.target_soft_updates)

    def get_stats(self, stats_sample):
        values = self.sess.run(self.stats_ops, feed_dict={
            self.obs0: stats_sample['obs0'],
            self.actions: stats_sample['actions'],
        })
        names = self.stats_names[:]
        assert len(names) == len(values)
        stats = dict(zip(names, values))
        return stats

    # Backup for parameter noise setting.
    # def _setup_param_noise(self, normalized_obs0):
    #     assert self.param_noise is not None
    #
    #     # Configure perturbed actor.
    #     param_noise_actor = copy(self.actor)
    #     param_noise_actor.name = 'param_noise_' + self.actor.name
    #     self.perturbed_actor_tf = param_noise_actor(normalized_obs0)
    #     logger.info('setting up param noise')
    #     self.perturb_policy_ops = self._get_perturbed_actor_updates(param_noise_actor)
    #
    #     # Configure separate copy for stddev adoption.
    #     adaptive_param_noise_actor = copy(self.actor)
    #     adaptive_param_noise_actor.name = 'adaptive_param_noise_' + self.actor.name
    #     adaptive_actor_tf = adaptive_param_noise_actor(normalized_obs0)
    #     self.perturb_adaptive_policy_ops = self._get_perturbed_actor_updates(adaptive_param_noise_actor)
    #     self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.actor_tf - adaptive_actor_tf)))
    #
    # def _get_perturbed_actor_updates(self, perturbed_actor):
    #     # assert len(actor.vars) == len(perturbed_actor.vars)
    #     # assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)
    #
    #     updates = []
    #     for var, perturbed_var in zip(self.actor.vars, perturbed_actor.vars):
    #         if var in self.actor.perturbable_vars:
    #             logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
    #             updates.append(
    #                 tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0.,
    #                                                                 stddev=self.param_noise_stddev)))
    #         else:
    #             logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
    #             updates.append(tf.assign(perturbed_var, var))
    #     assert len(updates) == len(self.actor.vars)
    #     return tf.group(*updates)

    # def adapt_param_noise(self):
    #     if self.param_noise is None:
    #         return 0.
    #
    #     # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
    #     batch = self.memory.sample(batch_size=self.batch_size)
    #     self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
    #         self.param_noise_stddev: self.param_noise.current_stddev,
    #     })
    #     distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
    #         self.obs0: batch['obs0'],
    #         self.param_noise_stddev: self.param_noise.current_stddev,
    #     })
    #
    #     mean_distance = np.mean(distance)
    #     self.param_noise.adapt(mean_distance)
    #     return mean_distance



