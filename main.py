"""
This file is the entrance of the nc-ddpg repository, which describe the run process in the most general(top) level.

"""
###
import argparse
###
def parse_args():
    """
    parse arguments from commandline.
    ------
    return: dictionary or argparse.Namespace, args stores arguments.
    """
    return args


def sample_trajectories(actor, replay_buffer):
    """
    collect Quads <state, action, reward, next_state> into replay_buffer by interacting with envionment with actor.
    ------
    parameters:
    actor:
    replay buffer:
    """
    pass

def update_paramter(actor, critic, target_actor, target_critic, replay_buffer):
    """
    update parameters of actor, critic, target_actor, target_critic with replay_buffer.
    ------
    parameters:
    actor:
    :param critic:
    :param target_actor:
    :param target_critic:
    :param replay_buffer:
    :return:
    """

if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # instantiate key components of DDPG algorithm: actor, critic, target actor, target_critic, replay_buffer.
    ###
    import Actor, Critic, ReplayBuffer
    ###
    actor = Actor(args)
    critic = Critic(args)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    replay_buffer = ReplayBuffer(args)

    # run the main process of DDPG algorithm: use actor to sample trajectories -> replay_buffer -> update actor/critic/target actor/target critic
    for i in range(args.epoch):
        sample_trajectories(actor, replay_buffer)
        update_parameter(actor, critic, target_actor, target_critic, replay_buffer)







