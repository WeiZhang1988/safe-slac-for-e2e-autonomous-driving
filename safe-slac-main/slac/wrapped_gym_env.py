import gym
from gym.spaces.box import Box
import numpy as np
from PIL import Image
from slac import carla_rl_env

class WrappedGymEnv(gym.Wrapper):
    def __init__(self, env, **kargs):
        super(WrappedGymEnv,self).__init__(env)
        self.task_name=kargs['task_name']
        self.height=kargs['image_size']
        self.width=kargs['image_size']
        self.action_repeat=kargs['action_repeat']
        self._max_episode_steps = 500
        self.observation_space=Box(0, 255, (3,self.height,self.width), np.uint8)
        self.action_space = env.action_space
        self.env=env
    def reset(self):
        img_np = self.env.reset()
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        return  np.transpose(img_np_resized,[2,0,1])
    def step(self, action):
        for _ in range(self.action_repeat):
            re = self.env.step(action)
        re=list(re)
        img_np = re[0]
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        re[0] = np.transpose(img_np_resized,[2,0,1])
        return tuple(re)

        
class WrappedGymEnv2(gym.Wrapper):
    def __init__(self, env, **kargs):
        super(WrappedGymEnv2,self).__init__(env)
        self.task_name=kargs['task_name']
        self.height=kargs['image_size']
        self.width=kargs['image_size']
        self.action_repeat=kargs['action_repeat']
        self._max_episode_steps = 1000
        self.observation_space=Box(0, 255, (6,self.height,self.width), np.uint8)
        self.ometer_space=Box(-np.inf, np.inf, shape=(40,2), dtype=np.float32)
        self.tgt_state_space=Box(0, 255, (3,self.height,self.width), np.uint8)
        self.action_space = Box(-1.0, 1.0, shape=(2,))
        self.env=env
    def reset(self):
        reset_output = self.env.reset()
        img_np = reset_output['front_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_1 = np.transpose(img_np_resized,[2,1,0])
        img_np = reset_output['lidar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_2 = np.transpose(img_np_resized,[2,1,0])
        src_img = np.concatenate((src_img_1, src_img_2), axis=0)
        img_np = reset_output['bev']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized,[2,0,1])
        wpsh = reset_output['wp_hrz']
        return  src_img, wpsh, tgt_img
    def step(self, action):
        if action[0] > 0:
            throttle = np.clip(action[0],0.0,1.0)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-action[0],0.0,1.0)
        act_tuple = ([throttle, brake, action[1]], [False])
        for _ in range(self.action_repeat):
            re = self.env.step(act_tuple)
        re=list(re)
        img_np = re[0]['front_camera']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_1 = np.transpose(img_np_resized,[2,1,0])
        img_np = re[0]['lidar_image']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        src_img_2 = np.transpose(img_np_resized,[2,1,0])
        src_img = np.concatenate((src_img_1, src_img_2), axis=0)
        img_np = re[0]['bev']
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize((self.height,self.width))
        img_np_resized = np.uint8(img_pil_resized)
        tgt_img = np.transpose(img_np_resized,[2,0,1])
        wpsh = re[0]['wp_hrz']
        return src_img, wpsh, tgt_img, re[1], re[2], re[3]
        
    def pid_sample(self):
        return self.env.pid_sample()
        
