import random
import os
from glob import glob
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


class RockPaperScissorsEnv(gym.Env):

    def __init__(self, env_config: dict[str, np.array]):

        super().__init__()

        self.rock = env_config['rock'].astype(np.uint8)
        self.scissors = env_config['scissors'].astype(np.uint8)
        self.paper = env_config['paper'].astype(np.uint8)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low = 0, high = 1.0, shape = (20, 20, 1), dtype = np.float32)
        self.total_wins = 0
        self.total_losses = 0
        self.n_rounds = 3 ## Number of rounds per episode
        self.current_step = -1
        self.total_score = 0

        self.observations = np.concatenate((self.rock, self.scissors, self.paper), axis = 0, dtype = np.uint8)
        self.states = np.array(len(self.rock) * [0] + len(self.scissors) * [1] + len(self.paper) * [2])
        self.n = len(self.observations)
        indices = list(range(self.n))
        random.shuffle(indices)
        self.observations = self.observations[indices]
        self.states = self.states[indices]

        self.p_fliplr = self.p_flipud = self.p_rot = 0.5
        self.wn = {'mean': 0, 'std': 255 * 0.05}

        assert len(self.observations) == len(self.states), 'E: Inconsistent lengths'

    def select_state(self):

        i = random.randint(0, self.n - 1)
        self.observation = self.observations[i]
        self.state = self.states[i]

    def preprocess_image(self):

        ## Flips image
        if random.random() <= self.p_fliplr:
            self.observation = np.fliplr(self.observation)
        if random.random() <= self.p_flipud:
            self.observation = np.flipud(self.observation)

        ## White noise
        noise = np.random.normal(loc = self.wn['mean'], scale = self.wn['std'], size = self.observation.shape).astype(int)
        self.observation = ((self.observation.astype(int) + noise) % 255).astype(np.uint8)

        ## Rotate image
        if random.random() <= self.p_rot:
            self.observation = np.rot90(m = self.observation, k = random.randint(1, 3), axes = (0, 1))

        ## Grayscale
        self.observation = cv2.cvtColor(self.observation, cv2.COLOR_RGB2GRAY)

        ## Normalize
        self.observation = (self.observation / 255.0)

        print(self.observation.shape)

        ## Resize
        self.observation = cv2.resize(self.observation, (self.observation_space.shape[:-1]))

        # self.observation = np.transpose(self.observation, axes = (2, 0, 1))#[..., np.newaxis]
        self.observation = self.observation[..., np.newaxis]

    def step(self, action):

        self.current_step += 1

        if action == self.state:
            reward = 1  # Tie
        elif (action == 0 and self.state == 1) or (action == 1 and self.state == 2) or (action == 2 and self.state == 0):
            reward = 2  # Win
        else:
            reward = -1  # Loss

        self.total_score += reward

        if self.current_step == self.n_rounds - 1:

            done = True

        else:

            done = False

        return self.observation, reward, done, False, {}

    def reset(self, seed=None, options=None):

        self.current_step = -1
        self.total_score = 0
        self.select_state()
        self.preprocess_image()

        return self.observation, {}

## Shuffle all lists
def shuffle(imgs, gts):

    indices = list(range(len(imgs)))
    random.shuffle(indices)

    imgs = imgs[indices]
    gts = gts[indices]

    return imgs, gts

def preprocess_image(img):

    ## Grayscale
    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)

    ## Normalize
    img = (img / 255.0)

    ## Resize
    img = cv2.resize(img, (20, 20))

    img = img[..., np.newaxis]

    return img

classes = ('rock', 'scissors', 'paper')
classes_distr = {class_name: None for class_name in classes}
classes_test_distr = {class_name: None for class_name in classes}
classes_train_distr = {class_name: None for class_name in classes}

dp = '../../archive'
img_fps = []
imgs = []
gts = []
test_ratio = 0.2
imgs_train = []
imgs_test = []
gts_train = []
gts_test = []

for class_idx, class_name in enumerate(classes):

    img_fps_batch = glob(os.path.join(dp, class_name, '*.png'))
    random.shuffle(img_fps_batch)
    img_fps = img_fps + img_fps_batch

    imgs_batch = [np.array(Image.open(img_fp).convert('RGB')) for img_fp in img_fps_batch]
    imgs = imgs + imgs_batch
    gts = gts + (len(img_fps_batch) * [class_idx])

    imgs_test_batch = imgs_batch[:int(test_ratio * len(img_fps_batch))]
    imgs_test = imgs_test + imgs_test_batch
    gts_test = gts_test + len(imgs_test_batch) * [class_idx]

    imgs_train_batch = imgs_batch[int(test_ratio * len(img_fps_batch)):]
    imgs_train = imgs_train + imgs_train_batch
    gts_train = gts_train + len(imgs_train_batch) * [class_idx]

    classes_distr[class_name] = np.array(imgs_batch)
    classes_test_distr[class_name] = np.array(imgs_test_batch)
    classes_train_distr[class_name] = np.array(imgs_train_batch)

imgs = np.array(imgs)
gts = np.array(gts)

imgs_test = np.array(imgs_test)
gts_test = np.array(gts_test)

imgs_train = np.array(imgs_train)
gts_train = np.array(gts_train)

n_instances = len(imgs)
n_test = len(imgs_test)
n_train = len(imgs_train)
print('Total number of instances:', n_instances)
print('Number of test instances:', n_test)
print('Number of train instances:', n_train)

# print(len(imgs), len(gts), '\n', len(imgs_test), len(gts_test), '\n', len(imgs_train), len(gts_train), '\n', len(imgs), len(imgs_test) + len(imgs_train))

assert len(imgs) == len(gts) and len(imgs_test) == len(gts_test) and len(imgs_train) == len(gts_train) and len(imgs) == len(imgs_test) + len(imgs_train), 'E: Inconsistent lengths'


imgs, gts = shuffle(imgs, gts)
imgs_train, gts_train = shuffle(imgs_train, gts_train)
imgs_test, gts_test = shuffle(imgs_test, gts_test)

rl_actions = []
for _ in range(0, 3*800, 800):
    print('Class', classes[gts[_]])
    print('Shape', imgs[_].shape)
    rl_actions.append(imgs[_])
    plt.imshow(imgs[_])
    plt.show()
    print('---')

gym.envs.register(
    id = 'RockPaperScissorsEnv-v0',
    entry_point = '__main__:RockPaperScissorsEnv', ## entry_point='<module_fp>:<environment_class>'
    kwargs = {'env_config': classes_train_distr}
)

ray.init(log_to_driver=False)

config = PPOConfig().rollouts(num_rollout_workers=1).resources(num_gpus=1).environment(env = RockPaperScissorsEnv, env_config = classes_train_distr)
config.model['conv_filters'] = [[64, [5, 5], 1], [64, [5, 5], 1], [16, [20, 20], 1]]

# config.model['fcnet_hiddens'] = [64,]
config.use_critic = True
config.use_gae = True
config.lambda_ = 0.95
config.use_kl_loss = True
config.sgd_minibatch_size = 64
config.num_sgd_iter = 30
config.shuffle_sequences = True
config.vf_loss_coeff = 0.5
config.entropy_coeff = 0.001
config.clip_param = 0.2

algo = (config.build())

print(algo.get_policy().model)

## Source: https://docs.ray.io/en/latest/rllib/package_ref/env.html
average_rewards_per_iteration = []
train_iterations = 20

for i in range(train_iterations):
    log = algo.train()
    print(pretty_print(log))
    average_rewards = log['sampler_results']['episode_reward_mean']
    average_rewards_per_iteration.append(average_rewards)

    print(f'Iteration {i + 1}, Average Rewards: {average_rewards}')


plt.plot(average_rewards_per_iteration)
plt.title('RockPaperScissors')
plt.xlabel('Iterations')
plt.ylabel('Average Rewards')
plt.show()

print(pd.DataFrame(average_rewards_per_iteration, columns = ['average_rewards_per_iteration']))


single_step_acc = 0
for state, class_name in enumerate(classes_train_distr.keys()):
    imgs = classes_test_distr[class_name]
    for i in range(len(imgs)):
        img = imgs[i]
        prep_img = preprocess_image(img)
        action = algo.compute_single_action(prep_img)
        single_step_acc += int((action == 0 and state == 1) or (action == 1 and state == 2) or (action == 2 and state == 0))

single_step_acc /= n_test
print('Accuracy:', single_step_acc)

single_step_acc = 0
for state, class_name in enumerate(classes_test_distr.keys()):
    imgs = classes_test_distr[class_name]
    for i in range(len(imgs)):
        img = imgs[i]
        prep_img = preprocess_image(img)
        action = algo.compute_single_action(prep_img)
        single_step_acc += int((action == 0 and state == 1) or (action == 1 and state == 2) or (action == 2 and state == 0))

single_step_acc /= n_test
print('Accuracy:', single_step_acc)

for _ in range(10):
    i = random.randint(a = 0, b = len(imgs_test))
    img = imgs_test[i]
    prep_img = preprocess_image(img)
    action = algo.compute_single_action(prep_img)
    gt = gts_test[i]
    plt.imshow(img)
    plt.show()
    print('Player\'s turn:', classes[gt])
    print('Agent\'s turn:', classes[action])
    print('---')

small_test_sample_fp = glob('../../small_test_sample/*')
small_test_sample = []
for img_fp in small_test_sample_fp:
    img = np.array(Image.open(img_fp).convert('RGB'))
    small_test_sample.append(img)
    prep_img = preprocess_image(img)
    action = algo.compute_single_action(prep_img)
    plt.imshow(img)
    plt.show()
    print('Agent\'s turn:', classes[action])
    print('---')