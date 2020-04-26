import sys
from tqdm import tqdm
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from projected_sinkhorn import conjugate_sinkhorn, projected_sinkhorn
from projected_sinkhorn import wasserstein_cost, euclidean_cost

import numpy as np

def attack(X,y, net, epsilon=0.01, epsilon_iters=10, epsilon_factor=1.1, 
           p=1, kernel_size=5, maxiters=400, 
           alpha=0.1, xmin=0, xmax=1, normalize=lambda x: x, verbose=0, 
           regularization=1000, sinkhorn_maxiters=100, 
           ball='wasserstein', norm='l2', clamping=True, constrained_sinkhorn=True,
           multiply=False, training=False, l1_delta=0.01, use_tqdm=False, targeted=False): 
    grad_sign = -1 if targeted else 1
    batch_size = X.size(0)
    image_d = X.size(-1)
    nchannels = X.size(1)
    alpha /= nchannels
    epsilon = X.new_ones(batch_size)*epsilon
    epsilon_plan = X.new_ones(batch_size)*epsilon
    C = wasserstein_cost(X, p=p, kernel_size=kernel_size)
    normalization = X.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
    X_ = X.clone()

    X_best = X.clone()
    err_best = err = y.new_full(y.size(), False, dtype=torch.bool) if targeted else (net(normalize(X)).max(1)[1] != y)

    epsilon_best = epsilon.clone()
    
    if err.all():
        return X_best, err_best, epsilon_best

    t = 0
    progress = tqdm(range(maxiters)) if use_tqdm else range(maxiters)
    save_params = None
    for i in progress:
        X_noErr = X_[~err]
        X_noErr.requires_grad = True
        opt = optim.SGD([X_], lr=0.1)
        loss = nn.CrossEntropyLoss()(net(normalize(X_noErr)),y[~err])
        opt.zero_grad()
        loss.backward()

        with torch.no_grad(): 
            # take a step
            if norm == 'linfinity': 
                X_noErr += alpha*torch.sign(X_noErr.grad)
            elif norm == 'enhanced_linfinity': 
                X_noErr += alpha * torch.sign(X_noErr.grad) * normalization[~err]
            elif norm == 'l2': 
                X_noErr += (alpha*X_noErr.grad/(X_noErr.grad.view(X_noErr.size(0),-1).norm(dim=1).view(X_noErr.size(0),1,1,1)))
            elif norm == 'wasserstein': 
                sd_normalization = X_.view(batch_size,-1).sum(-1).view(batch_size,1,1,1)
                X_noErr = (conjugate_sinkhorn(X_.clone()/sd_normalization, 
                                               X_.grad, C, alpha, regularization, 
                                               verbose=verbose, maxiters=sinkhorn_maxiters
                                               )*sd_normalization)[~err]
            elif norm == 'dim':
                X_noErr /= 1.05
            elif norm == 'grad':
                grad = grad_sign * X_noErr.grad
                grad_norm = grad.view(grad.size(0), -1).norm(p=float('inf'), dim=1).view(grad.size(0), 1, 1, 1)
                normed_grad = grad / grad_norm
                X_noErr = X_noErr + alpha * normed_grad * normalization[~err]
            else: 
                raise ValueError("Unknown norm")

            X_noErr = torch.clamp(X_noErr/normalization[~err], max=60/(image_d**2)/nchannels) * normalization[~err]
            
            # project onto ball
            if ball == 'wasserstein': 
                X_noErr, save_params = projected_sinkhorn(X[~err].detach()/normalization[~err]+10e-16, 
                                          X_noErr.detach()/normalization[~err], 
                                          C,
                                          normalization[~err] if constrained_sinkhorn \
                                                  else X.new_zeros(normalization[~err].size()),
                                          epsilon_plan[~err],
                                          regularization, 
                                          verbose=verbose,
                                          training=training,
                                          l1_delta=l1_delta,
                                          init_params=save_params,
                                          maxiters=sinkhorn_maxiters)
                X_noErr *= normalization[~err]
                new_epsilon = epsilon.new_zeros((epsilon[~err]).size())
                epsilon[~err] = torch.max(epsilon_plan[~err], new_epsilon)
            elif ball == 'linfinity':
                X_ = torch.min(X_, X + epsilon.view(X.size(0), 1, 1,1))
                X_ = torch.max(X_, X - epsilon.view(X.size(0), 1, 1,1))
            elif ball == 'l2':
                X_noErr = X[~err] + (X_noErr-X[~err]).renorm(p=2, dim=0, maxnorm=epsilon_plan[~err][0].item())
                epsilon[~err] = epsilon_plan[~err]
            else:
                raise ValueError("Unknown ball")

            if clamping:
                X_noErr = torch.clamp(X_noErr, min=xmin, max=xmax)
            else:
                X_noErr = torch.clamp(X_noErr, min=xmin).renorm(p=float('inf'), dim=0, maxnorm=xmax)
            
            if not training and constrained_sinkhorn:
                # checking the change in L1 norm after projection
                new_normalization = X_noErr.view(X_noErr.size(0),-1).sum(-1).view(X_noErr.size(0),1,1,1)
                max_diff_pct = ((normalization[~err]-new_normalization).abs()/normalization[~err]).max()
                assert max_diff_pct < 0.015, max_diff_pct
            
            X_[~err] = X_noErr

            if targeted:
                X_best = X_.clone() 
            else:
                err = (net(normalize(X_)).max(1)[1] != y)
            err_rate = err.sum().item()/batch_size
            if err_rate > err_best.sum().item()/batch_size:
                X_best = X_.clone() 
                epsilon_best = epsilon.clone()
                # initialize duals from the last iteration
                if save_params is not None:
                    save_params['alpha'] = save_params['alpha'][~err[~err_best]]
                    save_params['beta'] = save_params['beta'][~err[~err_best]]
                    save_params['psi'] = save_params['psi'][~err[~err_best]]
                    save_params['phi'] = save_params['phi'][~err[~err_best]]
                err_best = err.clone()

            if verbose and t % verbose == 0:
                print(t, loss.item(), epsilon.mean().item(), err_rate)
            
            t += 1
            if use_tqdm and len(epsilon[err]) > 0:
                progress.set_description(desc=f'{(1-err_rate)*100:.1f} {epsilon[err].max().item():.3f}', refresh=True)
            if err.all() or t == maxiters: 
                break

            if t > 0 and t % epsilon_iters == 0: 
                if multiply:
                    epsilon_plan[~err] *= epsilon_factor
                else:
                    epsilon_plan[~err] += epsilon_factor

    epsilon_best[~err] = float('inf')
    return X_best, err_best, epsilon_best
