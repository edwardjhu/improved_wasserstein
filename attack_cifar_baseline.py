'''Attack a CIFAR-10 model with a Wasserstein adversary.'''

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


parser = argparse.ArgumentParser(description='Attack a CIFAR-10 model with a Wasserstein adversary.')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--checkpoint', required=True)
# Directories
parser.add_argument('--outdir', default='epsilons/', help='output dir')
parser.add_argument('--datadir', default='data/', help='output dir')
# Threat model
parser.add_argument('--norm', default='grad')
parser.add_argument('--ball', default='wasserstein')
parser.add_argument('--p', default=1, type=float, help='p-wasserstein distance')
parser.add_argument('--alpha', default=0.06, type=float, help='PGD step size')
# Sinkhorn projection
parser.add_argument('--reg', default=3000, type=float, help='entropy regularization')
# Attack schedule
parser.add_argument('--init-epsilon', default=0.001, type=float, help='initial epsilon')
parser.add_argument('--epsilon-iters', default=1, type=int, help='freq to ramp up epsilon')
parser.add_argument('--epsilon-factor', default=.001, type=float, help='factor to ramp up epsilon')
parser.add_argument('--maxiters', default=400, type=int, help='PGD num of steps')
# MISC
parser.add_argument('--override', action='store_true')
parser.add_argument('--unconstrained', action='store_true')
parser.add_argument('--no-clamping', action='store_true')
parser.add_argument('--preset', default='new_clamping')
args = parser.parse_args()

if not args.override:
    if args.norm == 'linfinity':
        args.alpha = 0.1
    elif args.norm == 'grad':
        args.alpha = 0.06
    elif args.norm == 'enhanced_linfinity':
        args.alpha = 0.04 
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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
mu = torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(-1).unsqueeze(-1).to(device)
std = torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(-1).unsqueeze(-1).to(device)
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std

testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model architecture is from pytorch-cifar submodule
print('==> Building model..')
net = get_model('CIFAR10', args.model)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

regularization = args.reg
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
save_name = os.path.join(args.outdir, '{}_reg_{}_p_{}_alpha_{}_norm_{}_ball_{}_{}-form_{}.txt'.format(
                args.checkpoint.split('/')[-1], regularization, args.p, 
                args.alpha, args.norm, args.ball,
                'old' if args.unconstrained else 'new',
                'linf-renorming' if args.no_clamping else 'clamping'))
# Load checkpoint.
print('==> Resuming from checkpoint..')
checkpoint = torch.load(args.checkpoint)
if ('net' in checkpoint):
    net.load_state_dict(checkpoint['net'])
else:
    net.load_state_dict(checkpoint['state_dict'][0])

# freeze parameters
for p in net.parameters(): 
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()

print('==> regularization set to {}'.format(regularization))
print('==> p set to {}'.format(args.p))

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total

def test_attack(): 
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_epsilons = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_pgd, _, epsilons = attack(torch.clamp(unnormalize(inputs),min=0), 
                                         targets, net, 
                                         normalize=normalize, 
                                         regularization=regularization, 
                                         p=args.p, 
                                         alpha=args.alpha, 
                                         norm = args.norm, 
                                         ball = args.ball, 
                                         epsilon_iters=args.epsilon_iters,
                                         epsilon = args.init_epsilon, 
                                         epsilon_factor=args.epsilon_factor,
                                         clamping=not args.no_clamping,
                                         use_tqdm=True,
                                         constrained_sinkhorn=not args.unconstrained,
                                         maxiters=args.maxiters)

        outputs_pgd = net(normalize(inputs_pgd))
        loss = criterion(outputs_pgd, targets)

        test_loss += loss.item()
        _, predicted = outputs_pgd.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        epsilons[predicted == targets] = -1
        all_epsilons.append(epsilons)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg epsilon: %.3f'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, torch.cat(all_epsilons).float().mean().item()))
        print('\n')

        acc = 100.*correct/total
    all_epsilons = torch.cat(all_epsilons) 
    with open(save_name, 'w') as f:
        f.write('index\tradius\n')
        for i in range(len(all_epsilons)):
            f.write(f'{i+1}\t{all_epsilons[i].item()}\n')


print('==> Evaluating model..')
test()
print('==> Attacking model..')
test_attack()
