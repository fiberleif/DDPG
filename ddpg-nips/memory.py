import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def set_batch(self, idxs, new_data):
        self.data[(self.start + idxs) % self.maxlen] = new_data

    def clear(self):
        self.start, self.length = 0, 0


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    @property
    def nb_entries(self):
        return len(self.observations0)

class MultiprocessMemory(object):
    def __init__(self, num_process, limit, action_shape, observation_shape):
        self.num_process = num_process
        self.limit = limit
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.observations0 = [RingBuffer(limit, shape=observation_shape) for _ in range(num_process)]
        self.Qs = [RingBuffer(limit, shape=(1,)) for _ in range(num_process)]
        self.actions = [RingBuffer(limit, shape=action_shape) for _ in range(num_process)]
        self.rewards = [RingBuffer(limit, shape=(1,)) for _ in range(num_process)]
        self.terminals = [RingBuffer(limit, shape=(1,)) for _ in range(num_process)]
        self.last_end = 0

    def prepare_sample(self):
        self._idx = np.array(range(self.nb_entries))
        np.random.shuffle(self._idx)

    def sample(self, batch_size):
        ## Note: Call sample() will return all episodes with flatten.
        ##       Always call clear() after sample() calls.
        obs, Qs, rewards, actions, terminals = [], [], [], [], []
        idx = self._idx[(self.last_end + np.array(range(batch_size))) % self.nb_entries]
        self.last_end += batch_size
        for i in range(self.num_process):
            obs.append(self.observations0[i].get_batch(idx))
            Qs.append(self.Qs[i].get_batch(idx))
            rewards.append(self.rewards[i].get_batch(idx))
            actions.append(self.actions[i].get_batch(idx))
            terminals.append(self.terminals[i].get_batch(idx))
        obs = np.array(obs).reshape(-1, *self.observation_shape)
        actions = np.array(actions).reshape(-1, *self.action_shape)
        Qs = np.array(Qs).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        terminals = np.array(terminals).reshape(-1, 1)
        return {'obs0': obs, 'Qs': Qs, 'actions': actions, 'rewards': rewards, 'terminals': terminals}

    def clear(self):
        self.last_end = 0
        for i in range(self.num_process):
            for item in [self.observations0, self.actions, self.rewards, self.terminals]:
                item[i].clear()

    def append(self, obs0, Q, action, reward, terminal):
        for i in range(self.num_process):
            self.observations0[i].append(obs0[i])
            self.Qs[i].append(Q[i])
            self.actions[i].append(action[i])
            self.rewards[i].append(reward[i])
            self.terminals[i].append(terminal[i])

    def calc_return(self, last_values, last_dones, gamma=0.99, lam=0.95):
        idx = np.array(range(self.nb_entries))
        for i in range(self.num_process):
            last_value, last_done = last_values[i], last_dones[i]
            rewards = self.rewards[i].get_batch(idx)
            terminals = self.terminals[i].get_batch(idx)
            Qs = self.Qs[i].get_batch(idx)
            lastgaelam = 0
            for t in reversed(idx):
                nextnonterminal = 1.0 - terminals[t]
                if t == self.nb_entries - 1:
                    nextvalues = last_value
                else:
                    nextvalues = Qs[t+1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - Qs[t]
                lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                rewards[t] = lastgaelam + Qs[t]
            self.rewards[i].set_batch(idx, rewards)

    @property
    def nb_entries(self):
        return len(self.observations0[0])