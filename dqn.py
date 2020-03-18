from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action

            # forward the state rep screenshot into the network
            # np.argmax
            # with torch.no_grad():
            x = self.forward(state)
            # action = np.argmax(Variable(x)).item()  # exploit
            action = np.argmax(x.cpu().detach().numpy())




        else:
            action = random.randrange(self.env.action_space.n)  # explore
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here


    # Source: Hands- On Reinforcement Learning for Games...

    q_vals = model.forward(state)
    q_nextVals = model.forward(next_state)
    q_nextStateVals = target_model.forward(next_state)

    q_val = q_vals.gather(1, action.unsqueeze(-1)).squeeze(-1)
    q_nextVal = q_nextStateVals.gather(1, torch.max(q_nextVals, 1)[0].unsqueeze(-1)).squeeze(-1)


    expected_q_val = reward + (1 - done) * gamma * q_nextVal

    loss = torch.MSE_Loss(q_val, expected_q_val.detach())

    return loss



class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Randomly sampling data with specific batch size from the buffer
        # TODO: ReplayBuffer.sample(), you just have to unpack the element in the deque
        #  Each element is holding 5 things: state, action, reward, next_state, done. And you just have to index and return accordingly

        # Sample buffer randomly

        # Right now buffer stores each of them
        # separate each to their respective variables
        # separate the batch size of tuples to their respective variables also of bach size

        # sample buffer randomly
        focus_batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*focus_batch)


        return state, action, reward, next_state, done



    def __len__(self):
        return len(self.buffer)
