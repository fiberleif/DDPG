import argparse
import tensorflow as tf

class BaseClass(object):
    """
    This is the abstract class for all policies.
    If you want to implement a new policy, inherit this.
    """

    """
    The basic class for further designing class, supporting the hyper-parameter can be directly parsed from program outer.
    """
    DEFAULTS = {"hyperparamter_name": None}

    def __init__(self):
        self.parse_params(self.DEFAULTS)

    def parse_params(self, defaults):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        for name, d in defaults.items():
            bool_mapper = lambda str: True if 'True' in str or 'true' in str else False
            mapper = bool_mapper if type(d) is bool else type(d)
            parser.add_argument('--'+name, type=mapper if d is not None else float, default=d)
        args, _ = parser.parse_known_args()
        dict_args = vars(args)
        for name, v in dict_args.items():
            setattr(self, name, v)

