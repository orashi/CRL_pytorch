import argparse
import os
import random
import torch
from math import log10
import scipy.stats as stats
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable, grad
from models.crl import *
from data.dataset import CreateFT3DLoader

parser = argparse.ArgumentParser()
parser.add_argument('--FT3D', type=str, default="/home/orashi/datasets/SFD", help='path to FlyingThings3D dataset')
parser.add_argument('--KITTI15', type=str, default="/home/orashi/datasets/???", help='path to KITTI 2015 dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--cut', type=int, default=1, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--optim', action='store_true', help='load optimizer\'s checkpoint')
parser.add_argument('--outf', default='.', help='folder to output images and models checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--baseGeni', type=int, default=2500, help='start base of pure pair L1 loss')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default=None, help='tensorboard env')
parser.add_argument('--advW', type=float, default=0.0001, help='adversarial weight, default=0.0001')
parser.add_argument('--gpW', type=float, default=10, help='gradient penalty weight')

opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
gen_iterations = opt.geni
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed setup                                  # !!!!!
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up ends

writer = SummaryWriter(log_dir=opt.env)

dataloader = CreateFT3DLoader(opt)
