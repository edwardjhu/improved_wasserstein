'''Train a MNIST model against a Wasserstein adversary.'''

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


parser = argparse.ArgumentParser(description='Train a MNIST model against a Wasserstein adversary.')
parser.add_argument('--model', default='lenet')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='epochs to train to')
parser.add_argument('--seed', default=0, type=int, help='random seed')
# Directories
parser.add_argument('--outdir', default='checkpoints/', help='output dir')
parser.add_argument('--datadir', default='data/', help='output dir')
# Threat model
parser.add_argument('--p', default=1, type=float, help='p-wasserstein distance')
parser.add_argument('--norm', default='linfinity')
parser.add_argument('--ball', default='wasserstein')
parser.add_argument('--alpha', default=0.1, type=float, help='PGD step size')
# Sinkhorn projection
parser.add_argument('--reg', default=1000, type=float, help='entropy regularization')
parser.add_argument('--L1D', default=0.1, type=float, help='max L1 delta')
# Attack schedule
parser.add_argument('--init-epsilon', default=0.1, type=float, help='initial epsilon')
parser.add_argument('--epsilon-iters', default=1, type=int, help='freq to ramp up epsilon')
parser.add_argument('--epsilon-factor', default=1.4, type=float, help='factor to ramp up epsilon')
parser.add_argument('--maxiters', default=10, type=int, help='PGD num of steps')
# MISC
parser.add_argument('--override', action='store_true')

args = parser.parse_args()

if not args.override:
    if args.norm == 'linfinity':
        args.init_epsilon = 0.1
        args.alpha = 0.1
    elif args.norm == 'grad':
        args.alpha = 0.06
    elif args.norm == 'enhanced_linfinity':
        args.alpha = 0.04


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_file = os.path.join(args.outdir, f'mnist_adv_{args.model}_lr_{args.lr}_reg_{args.reg}_p_{args.p}_alpha_{args.alpha}_norm_{args.norm}_ball_{args.ball}_L1D_{args.L1D}_epoch_{{}}.pth')

print("==> Checkpoint directory", args.outdir)
print("==> Saving to", checkpoint_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root=args.datadir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root=args.datadir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = get_model('MNIST', args.model)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test_nominal(epoch):
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

    # Save checkpoint.
    acc = 100.*correct/total


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.outdir), 'Error: no checkpoint directory found!'
    resume_file = os.path.join(args.outdir, args.resume)
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    test_nominal(start_epoch)
    checkpoint_file = os.path.join(args.outdir, f'mnist_adv_lr_{args.lr}_reg_{args.reg}_p_{args.p}_alpha_{args.alpha}_norm_{args.norm}_ball_{args.ball}_L1D_{args.L1D}_epoch_{{}}.pth')

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    nominal_correct = 0
    total_epsilon = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        net.eval()
        inputs_pgd, _, epsilons = attack(inputs, targets, net, 
                                         epsilon_factor=args.epsilon_factor,
                                         epsilon=args.init_epsilon,
                                         maxiters=args.maxiters,
                                         epsilon_iters=args.epsilon_iters, 
                                         p=args.p, 
                                         regularization=args.reg, 
                                         alpha=args.alpha, 
                                         norm=args.norm, 
                                         ball=args.ball,
                                         multiply=True)
        net.train()
        optimizer.zero_grad()
        outputs = net(inputs_pgd.detach())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad(): 
            outputs_nominal = net(inputs)
            _, predicted_nominal = outputs_nominal.max(1)
            nominal_correct += predicted_nominal.eq(targets).sum().item()

            train_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_epsilon += (epsilons < float("inf")).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Adv Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Eps: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                100.*nominal_correct/total, nominal_correct, total, total_epsilon/total))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    nominal_correct = 0
    total = 0
    total_epsilon = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_pgd, _, epsilons = attack(inputs, targets, net, 
                                         epsilon_factor=args.epsilon_factor, epsilon=args.init_epsilon, 
                                         maxiters=args.maxiters, epsilon_iters=args.epsilon_iters, 
                                         p=args.p, 
                                         regularization=args.reg,
                                         alpha=args.alpha, 
                                         norm=args.norm, 
                                         ball=args.ball,
                                         sinkhorn_maxiters=10,
                                         training=True,
                                         l1_delta=args.L1D,
                                         multiply=True)
        with torch.no_grad():
            outputs = net(inputs_pgd)
            loss = criterion(outputs, targets)

            outputs_nominal = net(inputs)
            _, predicted_nominal = outputs_nominal.max(1)
            nominal_correct += predicted_nominal.eq(targets).sum().item()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_epsilon += epsilons.sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Eps: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                100.*nominal_correct/total, nominal_correct, total, total_epsilon/total))
 
    if epoch % 10 == 0:
        # Save checkpoint.
        acc = 100.*correct/total
        avg_epsilon = total_epsilon/total
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'eps': avg_epsilon,
            'epoch': epoch,
        }
        if not os.path.isdir(args.outdir):
            os.mkdir(args.outdir)
        torch.save(state, checkpoint_file.format(epoch))


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    if (epoch+1) % 10 == 0:
        test(epoch)
