from model import SeqRosModel
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time,sys
import io
from PIL import Image
import itertools
import pickle

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.0001                     # learning rate
EPSILON = 0.8                   # greedy policy
GAMMA = 0.98                    # reward discount
TARGET_REPLACE_ITER = 1000      # target update frequency
MEMORY_CAPACITY = 4000
env = SeqRosModel()

N_ACTIONS = 8
N_CHANNEL = 15
N_INPUT = 1
INPUT_SIZE = 128
N_STATES = INPUT_SIZE * INPUT_SIZE * N_CHANNEL * N_INPUT
input_buffer = []

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

# np.random.seed(7)
# torch.manual_seed(7)

class CNN_Net(nn.Module):
    def __init__(self, ):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(N_CHANNEL * N_INPUT, 32, 5, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )                               #output (32x16x16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )                               #output (64x8x8)
        self.out = nn.Linear(64*8*8, N_ACTIONS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        if use_cuda:
            self.eval_net, self.target_net = CNN_Net().cuda(), CNN_Net().cuda()
        else:
            self.eval_net, self.target_net = CNN_Net(), CNN_Net()

        self.e_greedy = EPSILON
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = np.reshape(x, (-1,N_CHANNEL*N_INPUT,INPUT_SIZE,INPUT_SIZE))
        if use_cuda:
            x = Variable(torch.FloatTensor(x), 0).cuda()
        else:
            x = Variable(torch.FloatTensor(x), 0)

        if np.random.uniform() < self.e_greedy:   # greedy
            actions_value = self.eval_net.forward(x)
            if use_cuda:
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        s = s.flatten()
        s_ = s_.flatten()

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('Parameters updated')
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        if use_cuda:
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]).cuda())
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).cuda())
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda())
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]).cuda())
        else:
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
        b_s = b_s.view(-1,N_CHANNEL * N_INPUT,INPUT_SIZE,INPUT_SIZE)
        b_s_ = b_s_.view(-1,N_CHANNEL * N_INPUT,INPUT_SIZE,INPUT_SIZE)
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

dqn.eval_net.load_state_dict(torch.load('dqn_eval_net_pretrained.pkl', map_location=lambda storage, loc: storage))
dqn.target_net.load_state_dict(torch.load('dqn_eval_net_pretrained.pkl', map_location=lambda storage, loc: storage))


overall_step = 0

#n1: ABarppapp; n2: ABarppppa; n3: ABarppapa; n4: ABarpppap; n5: ABarppaap; n6: ABarpppaa

NEIGHBOR_CANDIDATE_1 = [['ABarppppa', 'ABarppapp'], ['ABarpppap', 'ABarppapp'],
                        ['ABarppppa', 'ABarppapp', 'ABarpppap'], ['ABarppapp', 'ABarppapa', 'ABarppppa']]
NEIGHBOR_CANDIDATE_2 = [['ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppaap'], ['ABarpppaa', 'ABarppapa'], ['ABarpppaa', 'ABarppaap'], 
                        ['ABarpppap', 'ABarppapa', 'ABarpppaa'], ['ABarpppap', 'ABarppapa', 'ABarppaap'], 
                        ['ABarpppaa', 'ABarppaap', 'ABarpppap'], ['ABarpppaa', 'ABarppaap', 'ABarppapa'],
                        ['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']]
subgoals = list(itertools.product(NEIGHBOR_CANDIDATE_1, NEIGHBOR_CANDIDATE_2))
subgoals.append((['ABarppppa', 'ABarppapp', 'ABarpppap', 'ABarppapa'], ['ABarpppap', 'ABarppapa', 'ABarpppaa', 'ABarppaap']))

movement_types = []
cpaaa_locations = []
target_locations = []


for i_episode in range(len(subgoals)):
    del input_buffer[:]
    sg = subgoals[i_episode]
    print('Current Subgoals:')
    for g in sg:
        print(g)
    s = env.reset(sg)
    for ni in range(N_INPUT):
        input_buffer.append(s)
    s_all = []
    for input in input_buffer:
        if s_all == []:
            s_all = input
        else:
            s_all = np.concatenate((s_all, input), axis=0)  #s_all shape:(?*128*128)

    ep_r = 0
    counter = 0
    sg_counter = 0
    while True:
        # env.render()
        a = dqn.choose_action(s_all)

        # take action
        s_, r, sg_done, done = env.step(a)
        del input_buffer[0]
        input_buffer.append(s_)
        s_all_ = []
        for input in input_buffer:
            if s_all_ == []:
                s_all_ = input
            else:
                s_all_ = np.concatenate((s_all_, input), axis=0)

        counter += 1
        ep_r += r

        if sg_done:
            sg_counter += 1

        if done:
            movement_types.append(env.movement_types)
            cpaaa_locations.append(env.ai_locations)
            target_locations.append(env.target_locations)
            print('Ep: ', i_episode, '| Ep_reward: ', round(ep_r, 2), '| Step: ', counter)
            break
        s_all = s_all_


with open('./saved_data/movement_index.pkl', 'wb') as f:
    pickle.dump(movement_types, f)

with open('./saved_data/cpaaa_locations.pkl', 'wb') as f:
    pickle.dump(cpaaa_locations, f)

with open('./saved_data/target_locations.pkl', 'wb') as f:
    pickle.dump(target_locations, f)