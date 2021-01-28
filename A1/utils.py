import torch

import argparse
<<<<<<< HEAD
# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda')  
=======
DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda') 
>>>>>>> 7c18dc5e5774917a2f8e98cfbbec9aa40e5bbc1a



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
