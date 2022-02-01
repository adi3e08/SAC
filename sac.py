import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from collections import deque
import random
from copy import deepcopy

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1, d):            
        self.buffer.append((o, a, r, o_1, d))
    
    def sample(self, batch_size):
        O, A, R, O_1, D = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(np.array(O), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(O_1), dtype=torch.float, device=self.device),\
               torch.tensor(np.array(D), dtype=torch.float, device=self.device)

    def __len__(self):
        return len(self.buffer)

# Critic Network
class Q_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

# Actor Network
class Pi_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu(y2)

        if deterministic:
            # Only used for evaluating policy at test time.
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_sigma = self.log_sigma(y2)
            log_sigma = torch.clamp(log_sigma,min=-20.0,max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample()
            if with_logprob:
                log_prob = dist.log_prob(x_t).sum(1)
                log_prob -= (2*(np.log(2) - x_t - F.softplus(-2*x_t))).sum(1)
            else:
                log_prob = None
            action = torch.tanh(x_t)

        return action, log_prob

def process_dmc_observation(time_step):
    """
    Function to parse observation dictionary returned by Deepmind Control Suite.
    """
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    return o_1, r, done

def process_observation(x, simulator):
    if simulator == "dm_control":
        o_1, r, done = process_dmc_observation(x)
        if r is None:
            return o_1
        else:
            return o_1, r, done
    elif simulator == "gym":
        if type(x) is np.ndarray:
            return x
        elif type(x) is tuple:
            o_1, r, done, info = x
            return o_1, r, done

# Soft Actor Critic Algorithm
class SAC:
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.arglist = parse_args()
        self.env = make_env(seed)
        if self.arglist.use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.actor = Pi_FC(self.env.state_size,self.env.action_size).to(self.device)

        self.critic_1 = Q_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.critic_target_1 = deepcopy(self.critic_1)       
        for param in self.critic_target_1.parameters():
            param.requires_grad = False

        self.critic_2 = Q_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.critic_target_2 = deepcopy(self.critic_2)       
        for param in self.critic_target_2.parameters():
            param.requires_grad = False

        # Temperature is learnt
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float, device=self.device, requires_grad=True)
        # set target entropy to -|A|
        self.target_entropy = - self.env.action_size

        self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

        self.actor_loss_fn =  torch.nn.MSELoss()
        self.critic_loss_fn_1 =  torch.nn.MSELoss()
        self.critic_loss_fn_2 =  torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.arglist.lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.arglist.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.arglist.lr)

        self.exp_dir = os.path.join("./log", self.arglist.exp_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        if os.path.exists("./log"):
            pass            
        else:
            os.mkdir("./log")
        os.mkdir(self.exp_dir)
        os.mkdir(os.path.join(self.tensorboard_dir))
        os.mkdir(self.model_dir)

    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for episode in range(self.arglist.episodes):
            o = process_observation(self.env.reset(), self.env.simulator)
            ep_r = 0
            while True:
                if self.replay_buffer.__len__() >= self.arglist.start_steps:
                    with torch.no_grad():
                        a, _ = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                    a = a.cpu().numpy()[0]
                else:
                    a = np.random.uniform(self.env.action_low, self.env.action_high, size=self.env.action_size)

                o_1, r, done = process_observation(self.env.step(a), self.env.simulator)

                if self.env.simulator == "dm_control":
                    terminal = 0 # deep mind control suite tasks are infinite horizon i.e don't have a terminal state
                elif self.env.simulator == "gym":
                    terminal = int(done) # open ai gym tasks have a terminal state 
                self.replay_buffer.push(o, a, r, o_1, terminal)

                ep_r += r
                o = o_1

                if self.replay_buffer.__len__() < self.arglist.replay_fill:
                    pass
                else :
                    O, A, R, O_1, D = self.replay_buffer.sample(self.arglist.batch_size)

                    q_value_1 = self.critic_1(O, A)
                    q_value_2 = self.critic_2(O, A)

                    with torch.no_grad():
                        # Target actions come from *current* policy
                        A_1, logp_A_1 = self.actor(O_1, False, True)

                        next_q_value_1 = self.critic_target_1(O_1, A_1)                                    
                        next_q_value_2 = self.critic_target_2(O_1, A_1)
                        next_q_value = torch.min(next_q_value_1, next_q_value_2)
                        expected_q_value = R + self.arglist.gamma * (next_q_value - torch.exp(self.log_alpha) * logp_A_1) * (1 - D)

                    critic_loss_1 = self.critic_loss_fn_1(q_value_1, expected_q_value)
                    self.critic_optimizer_1.zero_grad()
                    critic_loss_1.backward()
                    self.critic_optimizer_1.step()

                    critic_loss_2 = self.critic_loss_fn_2(q_value_2, expected_q_value)
                    self.critic_optimizer_2.zero_grad()
                    critic_loss_2.backward()
                    self.critic_optimizer_2.step()

                    for param_1, param_2 in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                        param_1.requires_grad = False
                        param_2.requires_grad = False

                    A_pi, logp_A_pi = self.actor(O, False, True)
                    q_value_pi_1 = self.critic_1(O, A_pi)
                    q_value_pi_2 = self.critic_2(O, A_pi)
                    q_value_pi = torch.min(q_value_pi_1, q_value_pi_2)

                    actor_loss = - torch.mean(q_value_pi - torch.exp(self.log_alpha).detach() * logp_A_pi)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    self.log_alpha_optimizer.zero_grad()
                    alpha_loss = (torch.exp(self.log_alpha) * (-logp_A_pi - self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.log_alpha_optimizer.step()

                    for param_1, param_2 in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                        param_1.requires_grad = True
                        param_2.requires_grad = True

                    self.soft_update(self.critic_target_1, self.critic_1, self.arglist.tau)
                    self.soft_update(self.critic_target_2, self.critic_2, self.arglist.tau)

                if done:
                    writer.add_scalar('train_ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    break   

    def eval(self, episodes):
        ep_r_list = []
        for episode in range(episodes):
            o = process_observation(self.env.reset(), self.env.simulator)
            ep_r = 0
            while True:
                with torch.no_grad():
                    a, _ = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0),True)
                a = a.cpu().numpy()[0] 
                o_1, r, done = process_observation(self.env.step(a), self.env.simulator)
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list    

def parse_args():
    parser = argparse.ArgumentParser("SAC")
    parser.add_argument("--exp-name", type=str, default="expt_sac_cartpole_swingup", help="name of experiment")
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="use gpu")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="actor learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--start-steps", type=int, default=int(1e4), help="start steps")
    parser.add_argument("--replay-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=256, help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=50, help="eval over _ episodes")
    return parser.parse_args()

def make_env(env_seed):
    
    # import gym
    # env = gym.make('BipedalWalker-v3')
    # env.seed(env_seed)
    # env.state_size = 24
    # env.action_size = 4
    # env.action_low = -1
    # env.action_high = 1
    # env.simulator = "gym"

    # from dm_control import suite
    # env = suite.load(domain_name="reacher", task_name="hard", task_kwargs={'random': env_seed})
    # env.state_size = 6
    # env.action_size = 2
    # env.action_low = -1
    # env.action_high = 1
    # env.simulator = "dm_control"

    from dm_control import suite
    env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'random': env_seed})
    env.state_size = 5
    env.action_size = 1
    env.action_low = -1
    env.action_high = 1
    env.simulator = "dm_control"
    
    return env

if __name__ == '__main__':
    sac = SAC(seed=0)
    sac.train()

