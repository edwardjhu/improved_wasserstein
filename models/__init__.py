from .resnet import *

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_model(dataset, arch):
    if dataset == 'MNIST':
        if arch == 'lenet':
            net = nn.Sequential(
                    nn.Conv2d(1, 16, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 4, stride=2, padding=1),
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(32*7*7,100),
                    nn.ReLU(),
                    nn.Linear(100, 10)
                )
            return net
        elif arch == 'lenet_32':
            net = nn.Sequential(
                    nn.Conv2d(1, 32, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(64*7*7,16*7*7),
                    nn.ReLU(),
                    nn.Linear(16*7*7,100),
                    nn.ReLU(),
                    nn.Linear(100, 10)
                )
            return net
    elif dataset == 'CIFAR10':
        if arch == 'resnet18':
            return ResNet18()
        elif arch == 'resnet34':
            return ResNet34()
    assert False, f'Unknown Dataset/Arch combo: {dataset}, {arch}'
