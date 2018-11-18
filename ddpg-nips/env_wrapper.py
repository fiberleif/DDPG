import gym
import numpy as np

balance_pos = {'femur_r': np.array([-0.55308777, -0.51710186,  0.65322247]),
 'pros_tibia_r': np.array([-0.15818479, -0.97166179,  0.17564435]),
 'pros_foot_r': np.array([-0.08365099, -0.99215679,  0.09288392]),
 'femur_l': np.array([-0.55308777, -0.51710186, -0.65322247]),
 'tibia_l': np.array([-0.15818479, -0.97166179, -0.17564435]),
 'talus_l': np.array([-0.08365099, -0.99215679, -0.09288392]),
 'calcn_l': np.array([-0.13097871, -0.98666875, -0.09658859]),
 'toes_l': np.array([ 0.05820437, -0.99346389, -0.09819242]),
 'torso': np.array([-0.77731677,  0.6291094 ,  0.        ]),
 'head': np.array([-0.08353876,  0.99650453,  0.        ])}

def cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]

class ObsProcessWrapper(gym.Wrapper):
    def __init__(self, env, add_feature=True, round=2, y_axis=False, old_version=False, **kwargs):
        self.add_feature = add_feature
        self.old_version = old_version
        self.round = round
        self.y_axis = y_axis
        self.total_step = 0
        self.pos_x = 0
        super(ObsProcessWrapper, self).__init__(env)

    def obs_process_r1(self, state_desc):
        res = []
        pelvis = None

        for body_part in ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l',
                          'calcn_l', 'toes_l', 'torso', 'head']:
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                relative_pos = [cur[i] - pelvis[i] for i in range(2)]
                relative_vel = [cur[i] - pelvis[i] for i in range(2, 4)]
                relative_acc = [cur[i] - pelvis[i] for i in range(4, 6)]
                cur[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)]
                # compute extra observation
                ta = cross_product(balance_pos[body_part], relative_pos) / np.linalg.norm(relative_pos)
                tb = cross_product(relative_acc, relative_pos) / np.linalg.norm(relative_pos)
                res += cur
                if self.add_feature:
                    res += [ta * tb]  # negative ta * tb tends to fail.

        for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]
        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        if self.old_version and self.round == 2:
            res += state_desc["target_vel"]
            res += state_desc['body_pos']['pelvis'][0:2]
            res += [float(self.total_step) / 1000]
        if not self.old_version and self.round == 1:
            res += [float(self.total_step) / 300]

        self.pos_x = state_desc["body_pos"]["pelvis"][0]
        return res

    def obs_process_r2(self, state_desc):
        res = []
        pelvis = None

        # calculate convert matrix
        target_vel = state_desc["target_vel"]
        theta = np.arctan2(target_vel[2], target_vel[0])
        cosine, sine = np.cos(theta), np.sin(theta)

        for body_part in ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l',
                          'calcn_l', 'toes_l', 'torso', 'head']:
            cur = []

            def convertion(vec):
                t0 = vec[0] * cosine + vec[2] * sine
                t1 = vec[1]
                return [t0, t1]
            def convertion_y(vec):
                t0 = -vec[0] * sine + vec[2] * cosine
                return [t0]
            def rot_convertion(rot_vec):
                t0 = rot_vec[2] * cosine + rot_vec[0] * sine
                return [t0]

            cur += convertion(state_desc["body_pos"][body_part])
            cur += convertion(state_desc["body_vel"][body_part])
            cur += convertion(state_desc["body_acc"][body_part])
            cur += rot_convertion(state_desc["body_pos_rot"][body_part])
            cur += rot_convertion(state_desc["body_vel_rot"][body_part])
            cur += rot_convertion(state_desc["body_acc_rot"][body_part])
            if self.y_axis or body_part == "pelvis": # in all cases, add y axis observation of pelvis
                cur += convertion_y(state_desc["body_pos"][body_part])
                cur += convertion_y(state_desc["body_vel"][body_part])
                cur += convertion_y(state_desc["body_acc"][body_part])
            if body_part == "pelvis":
                pelvis = cur
                res += cur
            else:
                relative_pos = [cur[i] - pelvis[i] for i in range(2)]
                relative_vel = [cur[i] - pelvis[i] for i in range(2, 4)]
                relative_acc = [cur[i] - pelvis[i] for i in range(4, 6)]
                cur[:2] = [cur[i] - pelvis[i] for i in range(2)] # relative position to pelvis
                cur[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)] # relative angle position to pelvis
                # compute extra observation
                ta = cross_product(balance_pos[body_part], relative_pos) / np.linalg.norm(relative_pos)
                tb = cross_product(relative_acc, relative_pos) / np.linalg.norm(relative_pos)
                res += cur
                if self.add_feature:
                    res += [ta * tb]  # negative ta * tb tends to fail.

        for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        relative_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(3)]
        res += convertion(relative_pos)
        res += convertion(state_desc["misc"]["mass_center_vel"])
        res += convertion(state_desc["misc"]["mass_center_acc"])
        # in all cases, add center y axis observation of center mass
        res += convertion_y(relative_pos)
        res += convertion_y(state_desc["misc"]["mass_center_vel"])
        res += convertion_y(state_desc["misc"]["mass_center_acc"])

        pelvis_vel = state_desc["body_vel"]["pelvis"]
        res += [convertion(target_vel)[0] - convertion(pelvis_vel)[0], convertion_y(target_vel)[0] - convertion_y(pelvis_vel)[0]]
        res += [float(self.total_step) / 1000]

        self.pos_x = state_desc["body_pos"]["pelvis"][0]
        return res

    def step(self, action):
        obs, r, done, info = self.env.step(action, project=False)
        if obs is not None:
            if self.old_version:
                obs = self.obs_process_r1(obs)
            else:
                obs = self.obs_process_r1(obs) if self.round == 1 else self.obs_process_r2(obs)
            info['pos_x'] = self.pos_x
        self.total_step += 1
        return obs, r, done, info

    def reset(self):
        self.total_step = 0
        obs = self.env.reset(project=False)
        if self.old_version:
            obs = self.obs_process_r1(obs)
        else:
            obs = self.obs_process_r1(obs) if self.round == 1 else self.obs_process_r2(obs)
        return obs

class RewardReshapeWrapper(gym.Wrapper):
    def __init__(self, env, bonus):
        super(RewardReshapeWrapper, self).__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        r += self.bonus
        return obs, r, done, info

    def reset(self):
        return self.env.reset()

class FinalObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FinalObsWrapper, self).__init__(env)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        # since SubprocVecEnv will automatically reset env after it is done,
        # we have to save the final obs with info
        return obs, r, done, {'obs': obs}

    def reset(self):
        return self.env.reset()

class SkipframeWrapper(gym.Wrapper):
    def __init__(self, env, skipcnt):
        super(SkipframeWrapper, self).__init__(env)
        self.skipcnt = skipcnt

    def step(self, action):
        tr = 0.
        for i in range(self.skipcnt):
            obs, r, done, info = self.env.step(action)
            tr += r
            if done:
                break
        return obs, tr, done, info

    def reset(self):
        return self.env.reset()