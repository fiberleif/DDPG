import tensorflow as tf
import tensorflow.contrib as tc


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', distribution=None, layer_norm=True, activation=tf.nn.relu, layer_num=2, layer_width=64):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.distribution = distribution
        self.activation = activation
        self.layer_num = layer_num
        self.layer_width = layer_width

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs

            for _ in range(self.layer_num):
                x = tf.layers.dense(x, self.layer_width)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = self.activation(x)
            
            if self.distribution is not None:
                self.pd, pi = self.distribution.pdfromlatent(x, init_scale=0.01)
                x = self.pd.sample()
            else:
                x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)

            x = (x + 1.0) / 2.0
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True, withaction=True, activation=tf.nn.relu, layer_num=2, layer_width=64):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.withaction = withaction
        self.activation = activation
        self.layer_num = layer_num
        self.layer_width = layer_width

    def __call__(self, obs, action=None, reuse=False):
        if self.withaction:
            assert action is not None
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            for _ in range(self.layer_num):
                if self.withaction:
                    x = tf.concat([x, action], axis=-1)
                x = tf.layers.dense(x, self.layer_width)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = self.activation(x)

            """
            x = tf.layers.dense(x, self.layer_width)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = self.activation(x)

            if self.withaction:
                x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, self.layer_width)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = self.activation(x)
            """

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars