# Soft Actor-Critic (SAC)
This repository contains a clean and minimal implementation of Soft Actor-Critic (SAC) algorithm in Pytorch, for continuous action spaces.

SAC is a state-of-the-art model-free RL algorithm for continuous action spaces. It adopts an off-policy actor-critic approach and uses stochastic policies. It uses the maximum entropy formulation to achieve better exploration.

You can find more details about how SAC works in my accompanying blog post [here](https://adi3e08.github.io/blog/sac/).

## Results
I trained SAC on a few continuous control tasks from [Deepmind Control Suite](https://github.com/deepmind/dm_control/tree/master/dm_control/suite). Results are below.

* Cartpole Swingup : Swing up and balance an unactuated pole by applying forces to a cart at its base.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_cartpole_swingup.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_cartpole_swingup.gif" width="30%"/>
</p>

* Reacher Hard : Control a two-link robotic arm to reach a randomized target location.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_reacher_hard.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_reacher_hard.gif" width="30%"/>
</p>

* Cheetah Run : Control a planar biped to run.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_cheetah_run.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_cheetah_run.gif" width="30%"/>
</p>

* Walker Run : Control a planar walker to run.
<p align="center">
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_walker_run.png" width="40%"/>
<img src="https://adi3e08.github.io/files/blog/soft-actor-critic/imgs/sac_walker_run.gif" width="30%"/>
</p>

## Requirements
- Python
- Numpy
- Pytorch
- Tensorboard
- Matplotlib
- Deepmind Control Suite

## Usage
To train SAC on Walker Run task, run,

    python sac.py --domain walker --task run --mode train --episodes 3000 --seed 0 

The data from this experiment will be stored in the folder "./log/walker_run/seed_0". This folder will contain two sub folders, (i) models : here model checkpoints will be stored and (ii) tensorboard : here tensorboard plots will be stored.

To evaluate SAC on Walker Run task, run,

    python sac.py --domain walker --task run --mode eval --episodes 3 --seed 100 --checkpoint ./log/walker_run/seed_0/models/3000.ckpt --render

## References
* Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018a. [Link](https://arxiv.org/abs/1801.01290)
* Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905, 2018b. [Link](https://arxiv.org/abs/1812.05905)
