"""
This file contains the definition of the type of key components: actor/critic/target_actor/target_critic.

"""

import tensorflow as tf
import  tensorflow.contrib as tc

class Model(object):
    """
    Base class for actor and critic
    """
    def __init__(self, name):
        self.name = name

    @property
    def get_vars(self):
        """
        get all the variables of the model.
        ------
        :return: "list", a list of nodes in the computation graph.
        """
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def get_trainable_vars(self):
        """
        get all the trainable variables of the model.
        -------
        :return: "list", a list of trainable nodes in the computation graph.
        """
        return  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def get_perturbable_vars(self):
        """
        get all the perturbable variables of the model.
        -------
        :return: "list", a list of perturbable variables in the computation graph.
        """
        return [v for v in self.get_trainable_vars() if "LayerNorm" not in v.name]

class Actor(Model):
    """
     Standard class for actor
    """
    def __init__(self, obs_dim, action_dim, layer_norm=True, name='actor'):
        super(Actor, self).__init__(name=name)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layer_norm = layer_norm

        with tf.variable_scope(self.name) as scope:
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), "obs_ph")
            x = tf.layers.dense(self.obs_ph, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.relu(x)

            x = tf.layers.dense(x, self.action_dim, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
            # map [-1, 1] to [0, 1]
            x = (x + 1) / 2
            self.action_var = x


class Critic(Model):
    """
    Standard class for critic
    """
    def __init__(self, obs_dim, action_dim, layer_norm=True, name='critic'):
        super(Actor, self).__init__(name=name)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layer_norm = layer_norm

    def __call__(self, actor_action_var, reuse=False):
        with tf.variable_scope(self.name) as scope:
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), "obs_ph")
            self.action_ph = tf.placeholder(tf.float32, (None, self.action_dim), "action_ph")
            if reuse:
                scope.reuse_variables()
                x = tf.layers.dense(self.obs_ph, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, actor_action_var], axis=-1)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                self.value_with_actor = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            else:
                x = tf.layers.dense(self.obs_ph, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, self.action_ph], axis=-1)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                self.value_without_actor = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))


class ActorCritic():
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.sess = tf.Session()