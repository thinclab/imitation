import os
import numpy as np
from gym import utils
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

class HalfCheetahEnvMdfdReward(HalfCheetahEnv):
    def __init__(self):
        super(HalfCheetahEnvMdfdReward, self).__init__()

    def _step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        torsoanglebefore = self.model.data.qpos[2, 0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        torsoangleafter = self.model.data.qpos[2, 0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward_flat = -np.square(torsoangleafter)
        # print('ctrl: {0} run: {1} torso: {2}'.format(reward_ctrl, reward_run, reward_flat))
        reward = reward_ctrl + reward_run + reward_flat
        done = False

        x_velocity = (xposafter - xposbefore) / self.dt
        info = {
            "x_position": xposafter,
            "x_velocity": x_velocity,
            "reward_run": reward_run,
            "reward_ctrl": reward_ctrl,
        }

        return ob, reward, done, info

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat[1:],
    #         self.model.data.qvel.flat,
    #     ])

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = 10