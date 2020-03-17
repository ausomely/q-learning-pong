from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

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
            with torch.no_grad():
                x = self.model.foward(state)
                # action = np.argmax(Variable(x)).item()  # exploit
                action = np.argmax(x.cpu().detach().numpy())




        else:
            action = random.randrange(self.env.action_space.n) # explore
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

    x = model.foward(state).gather(1, action.view(action.size(0), 1))
    # x = model.forward(state).gather(1, action.unsqueeze(1))
    # x = x.squeeze(1)
    x_plusOne = target_model.foward(next_state)
    max_x_plusOne = model.foward(x_plusOne, 1)[0]
    expected_x = reward + (1 - done) * gamma * max_x_plusOne


    loss = torch.MSELoss(x, expected_x)
    
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
        state = []
        action = []
        reward = []
        next_state = []
        done = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state.append(state)
            action.append(action)
            reward.append(reward)
            next_state.append(next_state)
            done.append(done)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
