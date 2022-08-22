import os
import argparse
import gym
import numpy as np
import pickle
from sac import SAC

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic RL Demo Generator')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--only_render', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--render', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--path',  type=str,
                    help='path to load trained policy from', required=True)
parser.add_argument('--dem_length', type=int, default=30, 
                    help='length of each demonstration in steps')
parser.add_argument('--dem_amount', type=int, default=1000,
                    help='amount of demonstrations to record')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000, metavar='N',
                    help='size of replay buffer (default: 10000000)')


args = parser.parse_args()

# ENVIRONMENT SETUP
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

agent = SAC(env.observation_space.shape[0], env.action_space, args)

agent.load_checkpoint(args.path, evaluate=True)

demo = []

for i in range(args.dem_amount):
    state = env.reset()
    traj = []
    
    for j in range(args.dem_length):
        action = agent.select_action(state)

        if not args.only_render:

            d = {}
            d['obs']=state[0:2]
            d['acs']=action
            traj.append(d)
            

        next_state, reward, done, _ = env.step(action)
        if args.render:
            env.render(mode = "human")

        state = next_state
        
    if not args.only_render:
        
        demo.append(traj)

if not args.only_render:

    if not os.path.exists('demos/'):
        os.makedirs('demos/')
    items = args.path.split('_')
    demo_path = "demos/sac_demo_{}_{}".format(args.env_name, items[-1]) + "_pomdp.pickle"    
    
    print('Saving demos to {}'.format(demo_path))

    with open(demo_path, 'wb') as f:
        pickle.dump(demo, f)
