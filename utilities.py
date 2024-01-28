import torch
import torch.nn.functional as F
import numpy as np

softmax_helper = lambda x: F.softmax(x, 1)

def torch_dice_fn_bce(pred, target): #pytorch tensors NCDHW # should ideally do some thresholding but this approx is fine
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection) / (m1.sum() + m2.sum())