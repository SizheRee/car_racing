import gym
import torch
import cv2
import numpy as np
import utils
import logging
logger = logging.getLogger(__name__)

SEED = 1
IMG_STACK = 4
ACTION_REPEAT = 8


class GoodWrapper(gym.Wrapper):
    """
    进一步封装gym的env
    """
    def __init__(self, env):
        super().__init__(env)
        self.env.action_space.seed(SEED)
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        # 此处有问题，貌似reset返回为字典类型，其第一元素才是rgb图
        img_rgb = self.env.reset()[0]
        self.debug_img = img_rgb # debug image

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_norm = img_gray / 128. - 1.
        self.stack = [img_norm] * IMG_STACK
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(ACTION_REPEAT):
            observition, reward, terminated, truncated, info = self.env.step(action)
            # 前进step返回的obs是rgb图，非字典
            img_rgb = observition
            self.debug_img = img_rgb # debug image
            logger.debug("reward: %f", reward)
            die = terminated or truncated
            # 奖励死亡，初始策略的惩罚死亡效果并不好，赛车会来回徘徊且速度很慢
            if die:
                logger.debug("DIE")
                reward += -1
            # 进入周围场地惩罚 0.05
            if np.mean(img_rgb[63:65, 47:50, 1]) > 150:
                logger.debug("OUT")
                reward -= 0.1
            total_reward += reward
            # 最近几次没有提升，停止
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                logger.debug("DONE")
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == IMG_STACK
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        # gray = np.dot(rgb[0][..., :], [0.299, 0.587, 0.114])
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # print(gray.shape)
        # cv2.imshow("asdas", gray)
        # cv2.waitKey(1)
        if norm:
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # 回顾前一百次的经验
        count = 0
        length = 100
        history = np.zeros(length)
        # 记录
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class Agent():
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], dtype=np.float32, copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = utils.Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward