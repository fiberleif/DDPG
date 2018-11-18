from abc import ABC, abstractmethod
from baselines import logger
import argparse
from noise import *
import tensorflow as tf

class BasePolicy(ABC):
    """
    This is the abstract class for all policies.
    If you want to implement a new policy, inherit this.
    """

    """
    Define policy-specific parameters here. They will be parsed and saved.
    """
    DEFAULTS = {'noise-type': 'none'}

    def __init__(self):
        self.parse_params(self.DEFAULTS)

    def parse_params(self, defaults):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for name, d in defaults.items():
            bool_mapper = lambda str: True if 'True' in str or 'true' in str else False
            mapper = bool_mapper if type(d) is bool else type(d)
            parser.add_argument('--'+name, type=mapper if d is not None else float, default=d)
        args, _ = parser.parse_known_args()
        logger.info(str(_) + " is ignored by current policy")
        dict_args = vars(args)
        for name, v in dict_args.items():
            setattr(self, name, v)

    def parse_noise(self, action_shape):
        self.action_noise = None
        self.param_noise = None
        nb_actions = action_shape[-1]
        for current_noise_type in self.noise_type.split('*'):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=self.pn_init, desired_action_stddev=float(stddev), final_steps=self.final_steps, final_desired=self.final_action_noise)
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                self.action_noise = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions), dec=self.ou_dec, final_steps=self.final_steps, final_sigma=self.final_action_noise) for _ in range(self.num_process)]
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    @abstractmethod
    def initialize(self, sess):
        """
        initialize training with session sess.
        """
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    @abstractmethod
    def pi(self, obs, apply_noise=True, compute_Q=True):
        """
        Function pi() is used to generate action.
        """
        pass

    @abstractmethod
    def store_transition(self, obs, Q, action, reward, obs1, terminal):
        """
        Function store_transition is used to save transition to memory.
        Note that everything in the input is stacked from multiple processes.
        """
        pass

    def before_train(self):
        """
        Prepare for training.
        This is called before a group of train() calls.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Do training for one batch.
        """
        pass

    def update_target_net(self):
        """
        Update targetnet, if needed.
        This is called after each train() calls.
        Don't implement this method if your policy don't need update at this time.
        """
        pass

    def after_train(self):
        """
        Update, if needed.
        This is called after a group of train() calls.
        Don't implement this method if your policy don't need update at this time.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Clear, reset noise or sth else.
        This is called after a group of train() calls and after_train().
        """
        pass

    def adapt_param_noise(self):
        """
        Adjust the scale for perturbation.
        If no parameter noise is included, don't implement this.
        """
        return 0.

    def get_stats(self):
        """
        Get training stats.
        """
        return {}

    @property
    @abstractmethod
    def on_policy(self):
        """
        This must be implemented.
        """
        raise NotImplementedError
