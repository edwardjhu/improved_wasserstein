'''Attack a MNIST model with a Wasserstein adversary.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import get_model
from utils import progress_bar
from pgd import attack


parser = argparse.ArgumentParser(description='Attack a MNIST model with a Wasserstein adversary.')
parser.add_argument('--model', default='lenet')
parser.add_argument('--checkpoint', required=True)
# Directories
parser.add_argument('--outdir', default='epsilons/', help='output dir')
parser.add_argument('--datadir', default='data/', help='output dir')
# Threat model
parser.add_argument('--binarize', action='store_true')
parser.add_argument('--norm', default='grad')
parser.add_argument('--ball', default='wasserstein')
parser.add_argument('--p', default=1, type=float, help='p-wasserstein distance')
parser.add_argument('--alpha', default=0.06, type=float, help='PGD step size')
# Sinkhorn projection
parser.add_argument('--reg', default=1000, type=float, help='entropy regularization')
parser.add_argument('--kernel', default=5, type=int, help='width of the local transport plan')
# Attack schedule
parser.add_argument('--init-epsilon', default=0.01, type=float, help='initial epsilon')
parser.add_argument('--epsilon-iters', default=1, type=int, help='freq to ramp up epsilon')
parser.add_argument('--epsilon-factor', default=.01, type=float, help='factor to ramp up epsilon')
parser.add_argument('--maxiters', default=200, type=int, help='PGD num of steps')
# MISC
parser.add_argument('--override', action='store_true')
parser.add_argument('--preset', default='new_clamping')
parser.add_argument('--unconstrained', action='store_true')
parser.add_argument('--no-clamping', action='store_true')

args = parser.parse_args()

if not args.override:
    if args.norm == 'linfinity':
        args.init_epsilon = 0.1
        args.alpha = 0.1
    elif args.norm == 'grad':
        args.alpha = 0.06
    elif args.norm == 'enhanced_linfinity':
        args.alpha = 0.04
    if 'binarize' in args.checkpoint:
        args.binarize = True
    if args.preset == 'new_clamping':
        args.unconstrained = False
        args.no_clamping = False
    elif args.preset == 'old_clamping':
        args.unconstrained = True
        args.no_clamping = False
    elif args.preset == 'old_linf':
        args.unconstrained = True
        args.no_clamping = True
    else:
        assert False, f'Unknown preset: {args.preset}'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root=args.datadir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.MNIST(root=args.datadir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=4, pin_memory=True)

# Model
print('==> Building model..')
net = get_model('MNIST', args.model)
net = net.to(device)

regularization = args.reg
print('==> regularization set to {}'.format(regularization))

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

save_name = os.path.join(args.outdir, '{}_reg_{}_p_{}_alpha_{}_norm_{}_ball_{}_{}-form_{}.txt'.format(
                args.checkpoint.split('/')[-1], regularization, args.p, 
                args.alpha, args.norm, args.ball,
                'old' if args.unconstrained else 'new',
                'linf-renorming' if args.no_clamping else 'clamping'))

print('==> loading model {}'.format(args.checkpoint))
print('==> saving epsilon to {}'.format(save_name))
d = torch.load(args.checkpoint)
if 'state_dict' in d: 
    net.load_state_dict(d['state_dict'][0])

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
elif 'robust' in args.checkpoint: 
    net.load_state_dict(d)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
else: 
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(d['net'])

criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.binarize: 
                inputs = (inputs >= 0.5).float()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

def test_attack(): 
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_epsilons = []
    succeed_epsilons = []
    L1_delta = []
    W_delta = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.binarize: 
            inputs = (inputs >= 0.5).float()

        inputs_pgd, _, epsilons = attack(torch.clamp(inputs,min=0), targets, net,  
                                         regularization=regularization,
                                         p=args.p, 
                                         alpha=args.alpha, 
                                         norm=args.norm, 
                                         ball=args.ball,
                                         epsilon_iters=args.epsilon_iters,
                                         epsilon_factor=args.epsilon_factor,
                                         epsilon=args.init_epsilon,
                                         maxiters=args.maxiters,
                                         kernel_size=args.kernel,
                                         use_tqdm=True,
                                         clamping=not args.no_clamping,
                                         constrained_sinkhorn=not args.unconstrained)
        
        outputs_pgd = net(inputs_pgd)
        loss = criterion(outputs_pgd, targets)

        test_loss += loss.item()
        _, predicted = outputs_pgd.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        epsilons[predicted == targets] = float('inf')
        which_correct = epsilons == float('inf')
        succeed_epsilons.append(epsilons[~which_correct])
        all_epsilons.append(epsilons)
        
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg epsilon: %.3f'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, torch.cat(succeed_epsilons).float().mean().item()))
        acc = 100.*correct/total
        
    all_epsilons = torch.cat(all_epsilons) 
    with open(save_name, 'w') as f:
        f.write('index\tradius\n')
        for i in range(len(all_epsilons)):
            f.write(f'{i+1}\t{all_epsilons[i].item()}\n')


test()
test_attack()
