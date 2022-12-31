import os
import argparse
from copy import deepcopy
from collections import deque
import math
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
import glob
import subprocess
import matplotlib.pyplot as plt

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1):            
        self.buffer.append((o, a, r, o_1))
    
    def sample(self, batch_size):
        O, A, R, O_1 = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(np.array(O), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(A), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(R), dtype=torch.float64, device=self.device),\
               torch.tensor(np.array(O_1), dtype=torch.float64, device=self.device)

    def __len__(self):
        return len(self.buffer)

# Critic network
class Q_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size+action_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(torch.cat((x,a),1)))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2).view(-1)        
        return y

# Actor network
class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu(y2)

        if deterministic:
            # used for evaluating policy
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

# Parse observation dictionary returned by Deepmind Control Suite
def process_observation(time_step):
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    
    return o_1, r, done

# Soft Actor-Critic algorithm
class SAC:
    def __init__(self, arglist):
        self.arglist = arglist
        
        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)
        
        self.env = suite.load(domain_name=self.arglist.domain, task_name=self.arglist.task, task_kwargs={'random': self.arglist.seed})
        obs_spec = self.env.observation_spec()
        action_spec = self.env.action_spec()
        self.obs_size = np.sum([math.prod(obs_spec[k].shape) for k in obs_spec])
        self.action_size = math.prod(action_spec.shape)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.actor = Pi_FC(self.obs_size,self.action_size).to(self.device)

        if self.arglist.mode == "train":
            self.critic_1 = Q_FC(self.obs_size,self.action_size).to(self.device)
            self.critic_target_1 = deepcopy(self.critic_1)       
            self.critic_loss_fn_1 =  torch.nn.MSELoss()

            self.critic_2 = Q_FC(self.obs_size,self.action_size).to(self.device)
            self.critic_target_2 = deepcopy(self.critic_2)       
            self.critic_loss_fn_2 =  torch.nn.MSELoss()

            # set target entropy to -|A|
            self.target_entropy = - self.action_size

            path = "./log/"+self.arglist.domain+"_"+self.arglist.task
            self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
            self.model_dir = os.path.join(self.exp_dir, "models")
            self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

            if self.arglist.resume:
                checkpoint = torch.load(os.path.join(self.model_dir,"backup.ckpt"))
                self.start_episode = checkpoint['episode'] + 1

                self.actor.load_state_dict(checkpoint['actor'])
                self.critic_1.load_state_dict(checkpoint['critic_1'])
                self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
                self.critic_2.load_state_dict(checkpoint['critic_2'])
                self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
                self.log_alpha = torch.tensor(checkpoint['log_alpha'].item(), dtype=torch.float64, device=self.device, requires_grad=True)

                self.replay_buffer = checkpoint['replay_buffer']

            else:
                self.start_episode = 0

                self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float64, device=self.device, requires_grad=True)

                self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)

                if not os.path.exists(path):
                    os.makedirs(path)
                os.mkdir(self.exp_dir)
                os.mkdir(self.tensorboard_dir)
                os.mkdir(self.model_dir)

            for param in self.critic_target_1.parameters():
                param.requires_grad = False
            
            for param in self.critic_target_2.parameters():
                param.requires_grad = False

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.arglist.lr)
            self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.arglist.lr)
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=self.arglist.lr)

            if self.arglist.resume:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer_1.load_state_dict(checkpoint['critic_optimizer_1'])
                self.critic_optimizer_2.load_state_dict(checkpoint['critic_optimizer_2'])
                self.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])

                print("Done loading checkpoint ...")

            self.train()

        elif self.arglist.mode == "eval":
            checkpoint = torch.load(self.arglist.checkpoint,map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            ep_r_list = self.eval(self.arglist.episodes,self.arglist.render,self.arglist.save_video)

    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_backup(self, episode):
        checkpoint = {'episode' : episode,\
                      'actor' : self.actor.state_dict(),\
                      'actor_optimizer': self.actor_optimizer.state_dict(),\
                      'critic_1' : self.critic_1.state_dict(),\
                      'critic_optimizer_1': self.critic_optimizer_1.state_dict(),\
                      'critic_2' : self.critic_2.state_dict(),\
                      'critic_optimizer_2': self.critic_optimizer_2.state_dict(),\
                      'critic_target_1' : self.critic_target_1.state_dict(),\
                      'critic_target_2' : self.critic_target_2.state_dict(),\
                      'log_alpha' : self.log_alpha.detach(),\
                      'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),\
                      'replay_buffer' : self.replay_buffer \
                      }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))

    def soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        for episode in range(self.start_episode,self.arglist.episodes):
            o,_,_ = process_observation(self.env.reset())
            ep_r = 0
            while True:
                if self.replay_buffer.__len__() >= self.arglist.start_steps:
                    with torch.no_grad():
                        a, _ = self.actor(torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0))
                    a = a.cpu().numpy()[0]
                else:
                    a = np.random.uniform(-1.0, 1.0, size=self.action_size)

                o_1, r, done = process_observation(self.env.step(a))

                self.replay_buffer.push(o, a, r, o_1)

                ep_r += r
                o = o_1

                if self.replay_buffer.__len__() >= self.arglist.replay_fill:
                    O, A, R, O_1 = self.replay_buffer.sample(self.arglist.batch_size)

                    q_value_1 = self.critic_1(O, A)
                    q_value_2 = self.critic_2(O, A)

                    with torch.no_grad():
                        # Target actions come from *current* policy
                        A_1, logp_A_1 = self.actor(O_1, False, True)

                        next_q_value_1 = self.critic_target_1(O_1, A_1)                                    
                        next_q_value_2 = self.critic_target_2(O_1, A_1)
                        next_q_value = torch.min(next_q_value_1, next_q_value_2)
                        expected_q_value = R + self.arglist.gamma * (next_q_value - torch.exp(self.log_alpha) * logp_A_1)

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
                    writer.add_scalar('ep_r', ep_r, episode)
                    with torch.no_grad():
                        writer.add_scalar('alpha',torch.exp(self.log_alpha).item(),episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        # Evaluate agent performance
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    if (episode % 250 == 0 or episode == self.arglist.episodes-1) and episode > self.start_episode:
                        self.save_backup(episode)
                    break   

    def eval(self, episodes, render=False, save_video=False):
        # Evaluate agent performance over several episodes

        if render and save_video: 
            t = 0
            folder = "./media/"+self.arglist.domain+"_"+self.arglist.task
            subprocess.call(["mkdir","-p",folder])

        ep_r_list = []
        for episode in range(episodes):
            if render:
                vid = None
            o,_,_ = process_observation(self.env.reset())
            ep_r = 0
            while True:
                with torch.no_grad():
                    a, _ = self.actor(torch.tensor(o, dtype=torch.float64, device=self.device).unsqueeze(0),True)
                a = a.cpu().numpy()[0] 
                o_1, r, done = process_observation(self.env.step(a))
                if render:
                    img = self.env.physics.render(height=240,width=240,camera_id=0)
                    if vid is None:
                        vid = plt.imshow(img)
                    else:
                        vid.set_data(img)
                    plt.axis('off')
                    plt.pause(0.01)
                    plt.draw()
                    if save_video:
                        plt.savefig(folder + "/file%04d.png" % t, bbox_inches='tight')
                        t += 1
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    if render:
                        print("Episode finished with total reward ",ep_r)
                        plt.pause(0.5)                    
                    break        
        if self.arglist.mode == "eval":
            print("Average return :",np.mean(ep_r_list))
            if save_video:
                os.chdir(folder)
                subprocess.call(['ffmpeg', '-i', 'file%04d.png','-r','10','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt', 'yuv420p','video.mp4'])
                for file_name in glob.glob("*.png"):
                    os.remove(file_name)
                subprocess.call(['ffmpeg','-i','video.mp4','video.gif'])
        
        return ep_r_list

def parse_args():
    parser = argparse.ArgumentParser("Soft Actor-Critic")
    # Common settings
    parser.add_argument("--domain", type=str, default="", help="cartpole / reacher")
    parser.add_argument("--task", type=str, default="", help="swingup / hard")
    parser.add_argument("--mode", type=str, default="", help="train or eval")
    parser.add_argument("--episodes", type=int, default=0, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Train settings
    parser.add_argument("--resume", action="store_true", default=False, help="resume training")
    parser.add_argument("--lr", type=float, default=3e-4, help="actor, critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update parameter")
    parser.add_argument("--start-steps", type=int, default=int(1e4), help="start steps")
    parser.add_argument("--replay-size", type=int, default=int(1e6), help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=int(1e4), help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes during training")
    parser.add_argument("--eval-over", type=int, default=50, help="each time eval over _ episodes")
    # Eval settings
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint")
    parser.add_argument("--render", action="store_true", default=False, help="render")
    parser.add_argument("--save-video", action="store_true", default=False, help="save video")

    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    sac = SAC(arglist)
