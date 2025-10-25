import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from locconfig import LocConfig
import ot


class SlicedEarthMoversDistance(nn.Module):
    def __init__(self, num_projections=100, reduction='mean', scaling=1.0, p=1, normalize=True, device='cuda') -> None:
        super().__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'none':
            self.reduction = torch.nn.Identity()
        elif reduction == 'sum':
            self.reduction = torch.sum
        self.num_proj = num_projections
        self.eps = 1e-6
        self.scaling = scaling
        self.p = p
        self.normalize = normalize

    def forward(self, X, Y, *args):
        batch_tuple = X.shape[:-2]
        flat_X = X.reshape(batch_tuple + (-1,))

        # If max is 0, add epsilon
        max_vals, max_inds = flat_X.max(dim=-1)
        should_max = max_vals[:,0] < self.eps
        flat_X[should_max,0,max_inds[should_max,0]] = self.eps
        X = torch.mean(X, dim=1, keepdim=True)

        x = X[0,0]
        y = Y[0,0]
        x_coords = torch.nonzero(x > 0).float() / self.scaling
        y_coords = torch.nonzero(y > 0).float() / self.scaling
        dists = []
        if self.normalize:
            loss, projections = ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0]/x.sum(), y[y>0]/y.sum(),p=self.p, n_projections=self.num_proj, log=True)
        else:
            loss, projections = ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0], y[y>0],p=self.p, n_projections=self.num_proj, log=True)
        projections = projections['projections']
        for x, y in zip(X[1:],Y[1:]):
            x = x[0]
            y = y[0]
            x_coords = torch.nonzero(x > 0).float() / self.scaling
            y_coords = torch.nonzero(y > 0).float() / self.scaling
            if self.normalize:
                loss += ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0]/x.sum(), y[y>0]/y.sum(),p=self.p, projections=projections)
            else:
                loss += ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0], y[y>0],p=self.p, projections=projections)
        return loss


class CoMLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, pred_img, truth_img, tx_truth_coords):
        mean_pred = pred_img.mean(axis=1)
        centers_of_mass = get_centers_of_mass(mean_pred)
        error = torch.linalg.norm(tx_truth_coords[:,0,1:] - centers_of_mass, axis=1)
        return error.mean()


def unravel_indices(indices: torch.LongTensor, shape, ) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def get_centers_of_mass(tensor):
#taken from:https://gitlab.liu.se/emibr12/wasp-secc/blob/cb02839115da475c2ad593064e3b9daf2531cac3/utils/tensor_utils.py    
    """
    Args:
        tensor (Tensor): Size (*,height,width)
    Returns:
        Tuple (Tensor): Tuple of two tensors of sizes (*)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    width = tensor.size(-1)
    height = tensor.size(-2)
    
    x_coord_im = torch.linspace(0,width,width).repeat(height,1).to(device)
    y_coord_im = torch.linspace(0,width,height).unsqueeze(0).transpose(0,1).repeat(1,width).to(device)
    
    x_mean = torch.mul(tensor,x_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),1e-10)
    y_mean = torch.mul(tensor,y_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),1e-10)
    
    return torch.stack((y_mean, x_mean)).T


class TiremMLP(torch.nn.Module):
    def __init__(self, num_features=[14,200], device='cuda', dropout=0.01, input_dropout=0.1) -> None:
        super().__init__()
        self.tirem_bias = nn.Parameter(torch.ones(1))
        self.layers = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(num_features[0], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[-1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[-1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], 1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.to(device)

    def forward(self, x, tirem_pred):
        #tirem_bounded = nn.functional.relu(tirem_pred + self.tirem_bias, inplace=True)
        return self.layers(x)# + tirem_bounded[:,None]
    

class TiremMLP2(torch.nn.Module):
    def __init__(self, num_features=[14,200, 100], device='cuda', dropout=0.01, input_dropout=0.1) -> None:
        super().__init__()
        in_features, hidden1, hidden2 = num_features
        self.layers11 = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(num_features[0], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[-1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.layers12 = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(num_features[0], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[-1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.layers2 = nn.Sequential(
            nn.Linear(num_features[-1]*2, num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[-1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[-1], 1),
        )
        self.to(device)

    def forward(self, x1, x2):
        x1 = self.layers11(x1)
        x2 = self.layers12(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.layers2(x)

        return x