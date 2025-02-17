import gym
from full_state_car_racing_env import FullStateCarRacingEnv
import scipy
import scipy.misc
import numpy as np
import argparse
import torch
import os
import imageio
from tqdm import tqdm
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

def run(steering_network, args):
    
    env = FullStateCarRacingEnv()
    env.reset()
    
    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    cross_track_error_h = 0
    cross_track_error_d = 0
    i = 0
    for t in range(args.timesteps):
        i = t
        env.render()
        
        state, expert_action, reward, done, _ = env.step(learner_action)
        e = env.get_cross_track_error(env.car, env.track)
        cross_track_error_h += abs(e[0])
        cross_track_error_d += abs(e[1])
        if done:
            break
        
        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]    # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]

        if args.expert_drives:
            learner_action[0] = expert_steer
        else:
            learner_action[0] = steering_network.eval1(state, device=DEVICE)

        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        if args.save_expert_actions:
            imageio.imsave(os.path.join(args.out_dir, 'expert_%d_%d_%f.jpg' % (args.run_id, t, expert_steer)), state)
    env.close()
    return cross_track_error_h / (i + 1), cross_track_error_d / (i + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", help="directory in which to save the expert's data", default='./dataset_1/train')
    parser.add_argument("--save_expert_actions", type=str2bool, help="save the images and expert actions in the training set",
                        default=False)
    
    parser.add_argument("--expert_drives", type=str2bool, help="should the expert steer the vehicle?", default=False)
    parser.add_argument("--run_id", type=int, help="Id for this particular data collection run (e.g. dagger iterations)", default=0)
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run, up to one full loop of the track", default=100000)
    parser.add_argument("--learner_weights", type=str, help="filename from which to load learner weights for the steering network",
                        default='')
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    
    args = parser.parse_args()
    
    steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    if args.learner_weights:
        steering_network.load_weights_from(args.learner_weights)

    run(steering_network, args)

