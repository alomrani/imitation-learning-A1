import train_policy
import racer
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
# from tqdm import tdqm
from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from train_policy import get_class_distribution, test_discrete, train_discrete
from utils import DEVICE, str2bool
import torch
from racer import run
   
def train_epochs(args, data_transform):
    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)
    
    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)
    
    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    
    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()

    args.class_dist = get_class_distribution(training_iterator, args)
    best_val_accuracy = 0 
    for epoch in range(args.n_epochs):
        print ('EPOCH ', epoch)
        
        train_discrete(driving_policy, training_iterator, opt, args)
        acc = test_discrete(driving_policy, validation_iterator, opt, args)
        if acc > best_val_accuracy:
            torch.save(driving_policy.state_dict(), args.weights_out_file)
            best_val_accuracy = acc
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='./weights')
    parser.add_argument("--dagger_iterations", help="", default=10)
    args = parser.parse_args()

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####
    print('TRAINING LEARNER ON INITIAL DATASET')
    
    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])
    

    train_epochs(args, data_transform)
    cr = []
    for i in range(args.dagger_interations):

        print('GETTING EXPERT DEMONSTRATIONS')
        current_learner = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        current_learner.load_weights_from(args.weights_out_file)
        cross_track_error = run(current_learner, args)
        cr.append(cross_track_error)
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        args.weights_out_file = os.path.join(args.weights_out_file, "learner_{i}_weights.weights")
        train_epochs(args, data_transform)