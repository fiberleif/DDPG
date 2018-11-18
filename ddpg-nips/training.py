import os
import time
from collections import deque
import pickle
import baselines.common.tf_util as U
from baselines import logger
import numpy as np
import tensorflow as tf
from subprocess import check_output

def train(env, agent, num_process, nb_epochs, nb_epoch_cycles, nb_train_steps, nb_rollout_steps, 
    nb_eval_steps, batch_size, memory, visualizer, HPC, model_dir, eval_env=None, **kwargs):

    nb_rollout_steps //= num_process
    nb_epoch_cycles *= num_process
    # nb_train_steps = nb_train_steps * num_process
    saver = tf.train.Saver()
    best_saver = tf.train.Saver()

    step, episode = 0, 0
    best_eval = -1000.
    start_time = time.time()
    episode_reward = np.zeros((num_process,), dtype=np.float64)
    episode_step = np.zeros((num_process,), dtype=np.int32)
    visualizer.initialize('train-reward', 'red')
    if eval_env is not None:
        visualizer.initialize('eval-reward', 'cyan')
        visualizer.initialize('distance*1000', 'green')

    with U.single_threaded_session() as sess:
        print('model_dir', model_dir)
        if not model_dir:
            agent.initialize(sess)
        else:
            new_saver = tf.train.import_meta_graph(model_dir + '.meta')
            new_saver.restore(sess, model_dir)
            agent.sess = sess
        tf.add_to_collection('eval', agent.obs0)
        tf.add_to_collection('eval', agent.actions)
        tf.add_to_collection('eval', agent.actor_tf)
        tf.add_to_collection('eval', agent.critic_tf)
        sess.graph.finalize()
        agent.reset()
        obs = env.reset()
        for epoch in range(nb_epochs):
            epoch_reward, epoch_steps, epoch_episode = [], [], 0
            epoch_aloss, epoch_closs, epoch_dis = [], [], []
            epoch_start_time = time.time()
            for cycle in range(nb_epoch_cycles):
                if agent.param_noise:
                    agent.param_noise.decay(steps=step * num_process)
                if agent.action_noise:
                    for noise in agent.action_noise:
                        noise.decay(steps=step * num_process)

                for t_rollout in range(nb_rollout_steps):
                    if agent.step_noise and agent.param_noise:
                        agent.sess.run(agent.perturb_policy_ops, feed_dict={agent.param_noise_stddev: agent.param_noise.current_stddev})
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)

                    new_obs, r, done, info = env.step(action)
                    obs_before_reset = [t['obs'] for t in info]
                    agent.store_transition(obs, q, action, r, obs_before_reset, done)
                    obs = new_obs
                    ###
                    episode_reward += r
                    episode_step += 1
                    step += 1
                    for s in range(num_process):
                        if done[s]:
                            epoch_reward.append(episode_reward[s])
                            epoch_steps.append(episode_step[s])
                            episode_step[s] = 0
                            episode_reward[s] = 0
                            episode += 1
                            epoch_episode += 1
                ##### train
                if memory.nb_entries >= batch_size:
                    if agent.on_policy:
                        agent.calc_return(agent.getq(obs), done)
                    agent.before_train()
                    for t_train in range(nb_train_steps):
                        distance = agent.adapt_param_noise()
                        epoch_dis.append(distance)
                        c1, a1 = agent.train()
                        agent.update_target_net()
                        epoch_closs.append(c1)
                        epoch_aloss.append(a1)
                    agent.after_train()
                    agent.reset()
            ##### eval
            timer = time.time()
            if eval_env is not None:
                eval_rewards = []
                eval_forward = []
                for eval_time in range(4):
                    eval_obs = eval_env.reset()
                    eval_episode_rewards = np.zeros(num_process)
                    eval_forward = [0 for _ in range(num_process)]
                    dones = [False for _ in range(num_process)]

                    for t_rollout in range(nb_eval_steps):
                        mask = 1 - np.array(dones).astype(np.float32)
                        action, q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, r, current_dones, infos = eval_env.step(action)
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
            ##### After an epoch, log
            is_best = False
            stats = agent.get_stats()
            stats['total/steps'] = step * num_process
            stats['total/episode'] = episode
            stats['total/epochs'] = epoch + 1
            stats['total/time'] = time.time() - start_time
            stats['epoch/reward_sum'] = np.mean(epoch_reward)
            stats['epoch/steps'] = np.mean(epoch_steps)
            stats['epoch/episode'] = epoch_episode
            stats['epoch/time'] = time.time() - epoch_start_time
            stats['train/actor_loss'] = np.mean(epoch_aloss)
            stats['train/critic_loss'] = np.mean(epoch_closs)
            stats['train/param_noise_distance'] = np.mean(epoch_dis)
            if eval_env is not None:
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

            ##### Visualize
            visualizer.paint('train-reward', stats['total/steps']/1e6, stats['epoch/reward_sum'])
            if eval_env is not None:
                visualizer.paint('eval-reward', stats['total/steps']/1e6, stats['eval/reward_sum'])
                visualizer.paint('distance*1000', stats['total/steps']/1e6, stats['eval/mean_forward'] * 1000)

            ##### Save model
            if saver is not None:
                saver.save(sess, os.path.join(logger.get_dir(), 'model'), global_step=epoch+1)
                if is_best:
                    best_saver.save(sess, os.path.join(logger.get_dir(), 'model-best'), global_step=epoch+1)
            if HPC and epoch % 5 == 0:
                try:
                    logger.info(check_output("python copyexist.py", shell=True))
                except:
                    pass