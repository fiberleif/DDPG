import numpy as np


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01, final_desired = 0.003, final_steps = 3e6):
        self.initial_stddev = initial_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

        self.init_desired = desired_action_stddev
        self.desired_action_stddev = desired_action_stddev
        self.final_desired = final_desired
        self.final_steps = final_steps

    def decay(self, steps):
        if steps >= self.final_steps:
            progress = 1
        else:
            progress = steps / self.final_steps
        progress = (-progress ** 2 + 2 * progress) ** 0.5 # decay faster in the beginning, slower in the end
        self.desired_action_stddev = (1 - progress) * self.init_desired + progress * self.final_desired
        # print('step:', steps, 'param action std:', self.desired_action_stddev)

    def adapt(self, distance):
        # print('param noise:', self.current_stddev, 'action std:', distance)
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, dec, theta=.15, dt=1e-2, x0=None, final_sigma=0.003, final_steps = 3e6):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.min_sigma = sigma/100
        self.dt = dt
        self.x0 = x0
        self.dec=dec
        self.init_sigma = sigma[0]
        self.final_sigma = final_sigma
        self.final_steps = final_steps
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        # print('ou std:', x.std())
        return x

    def decay(self, steps):
        if steps >= self.final_steps:
            progress = 1
        else:
            progress = steps / self.final_steps
        progress = (-progress ** 2 + 2 * progress) ** 0.5
        sigma = ((1 - progress) * self.init_sigma + progress * self.final_sigma)
        self.sigma = np.ones_like(self.sigma) * sigma
        # print('step:', steps, 'ou action std:', sigma)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        # if self.sigma[0]>self.min_sigma[0]:
        #     self.sigma -= self.dec

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
