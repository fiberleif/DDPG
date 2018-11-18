import numpy as np
import tensorflow as tf
from baselines import logger
import baselines.common.tf_util as U
from copy import copy
from functools import reduce
from util import reduce_std, Adam, RunningMeanStd
from .basepolicy import BasePolicy
from .utils import get_target_updates

class PPO(BasePolicy):
    DEFAULTS = {
            'run-name': 'PPOtest',
            'layer-norm': True,
            #'normalize-returns': False,
            #'normalize-observations': True,
            #'critic-l2-reg': 1e-2,
            'lr': 3e-4,
            'critic-coef': 0.5,
            'entropy-coef': 0.001,
            'clip-norm': 0.5,
            'batch-size': 64,
            'nb-train-steps': 8,
            'nb-rollout-steps': 128,
            'gamma': 0.99,
            'lam': 0.95,
            'epsilon': 0.2,
            'reward-scale': 1.0,
            'noise-type': 'none'
        }
    
    def __init__(self, actor, critic, memory, observation_shape, action_shape):
        super(PPO, self).__init__()

        self.obs0 = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        self.returns = tf.placeholder(tf.float32, shape=(None, 1), name='return')
        self.actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='action')
        
        self.actor = actor
        self.critic = critic
        self.memory = memory

        ### TODO: normalize observations
        self.obs_rms = None
        normalized_obs0 = self.obs0

        target_actor = copy(actor)
        target_actor.name = 'target_actor'
        self.target_actor = target_actor
        target_critic = copy(critic)
        target_critic.name = 'target_critic'
        self.target_critic = target_critic

        self.actor_tf = actor(normalized_obs0)
        self.neglogp_act = actor.pd.neglogp(self.actions * 2 - 1) # Temporary solution
        self.target_actor_tf = target_actor(normalized_obs0)
        self.neglogp_target = target_actor.pd.neglogp(self.actions * 2 - 1) # 
        self.entropy = tf.reduce_mean(actor.pd.entropy())

        self.critic_tf = critic(normalized_obs0)
        self.target_critic_tf = target_critic(normalized_obs0)
        self.adv = self.returns - self.critic_tf
        self.targetadv = self.returns - self.target_critic_tf
        advmean, advvar = tf.nn.moments(self.targetadv, [0])
        self.targetadv = tf.squeeze(tf.nn.batch_normalization(self.targetadv, advmean, advvar, None, None, 1e-5))
        self.clipadv = self.returns - (self.target_critic_tf + 
            tf.clip_by_value(self.critic_tf - self.target_critic_tf, -self.epsilon, self.epsilon))
        self.ratio = tf.exp(self.neglogp_target - self.neglogp_act)
        self.clipratio = tf.clip_by_value(self.ratio, 1-self.epsilon, 1+self.epsilon)

        self.setup_optimizer()
        self.setup_target_network_updates()

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, 1.0)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, 1.0)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def setup_optimizer(self):
        logger.info('setting up optimizer: learning rate ' + str(self.lr))
        self.see = (tf.maximum(self.ratio * -self.targetadv, self.clipratio * -self.targetadv))
        self.actor_loss = tf.reduce_mean(tf.maximum(self.ratio * -self.targetadv, self.clipratio * -self.targetadv))
        self.critic_loss = 0.5 * tf.reduce_mean(tf.maximum(tf.square(self.adv), tf.square(self.clipadv)))
        self.loss = self.actor_loss + self.critic_loss * self.critic_coef - self.entropy * self.entropy_coef
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        logger.info('  actor shapes: {}'.format(actor_shapes))
        logger.info('  actor params: {}'.format(actor_nb_params))
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        logger.info('  critic shapes: {}'.format(critic_shapes))
        logger.info('  critic params: {}'.format(critic_nb_params))
        self.grads = U.flatgrad(self.loss, self.actor.trainable_vars + self.critic.trainable_vars, clip_norm=self.clip_norm)
        self.optimizer = Adam(var_list=self.actor.trainable_vars + self.critic.trainable_vars,
            beta1=0.9, beta2=0.999, epsilon=1e-08)

    def pi(self, obs, apply_noise=True, compute_Q=True):
        feed_dict = {self.obs0: obs}
        if compute_Q:
            action, q = self.sess.run([self.target_actor_tf, self.target_critic_tf], feed_dict=feed_dict)
        else:
            action = self.sess.run(self.target_actor_tf, feed_dict=feed_dict)
            q = None
        return action, q

    def getq(self, obs):
        feed_dict = {self.obs0: obs}
        q = self.sess.run(self.target_critic_tf, feed_dict=feed_dict)
        return q

    def store_transition(self, obs, Q, action, reward, obs1, terminal):
        self.memory.append(obs, Q, action, reward * self.reward_scale, terminal)
        ### TODO: update normalize observation

    def train(self):
        batch = self.memory.sample(self.batch_size)
        ops = [self.actor_loss, self.grads, self.critic_loss]
        actor_loss, grads, critic_loss = self.sess.run(ops, feed_dict={
                self.obs0: batch['obs0'],
                self.actions: batch['actions'],
                self.returns: batch['rewards']
            })
        self.optimizer.update(grads, stepsize=self.lr)
        #print(critic_loss, actor_loss)
        return critic_loss, actor_loss

    def initialize(self, sess):
        super(PPO, self).initialize(sess)
        self.optimizer.sync()
        self.sess.run(self.target_init_updates)

    def get_stats(self):
        ## TODO: collect stats
        stats = dict()
        return stats

    def calc_return(self, last_obs, last_done):
        self.memory.calc_return(last_obs, last_done, self.gamma, self.lam)

    def reset(self):
        self.memory.clear()

    def after_train(self):
        self.sess.run(self.target_soft_updates)

    def before_train(self):
        self.memory.prepare_sample()

    @property
    def on_policy(self):
        return True
    