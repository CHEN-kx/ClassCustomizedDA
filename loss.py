import numpy as np

import torch
import torch.nn as nn

def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    return (XX + YY - XY -YX)

def DANNLoss_diffsize(features, ad_net, sub_index, use_target):
    ''' Adversarial Loss for source(1) and target(0) '''
    batch_size = features.size(0) // 2
    pos_feature = features[:batch_size, :][sub_index]
    if not use_target:
        neg_feature = features[:batch_size, :][~sub_index]
    else:
        neg_feature = features[batch_size:, :]
    
    ad_out = ad_net(torch.cat((pos_feature,neg_feature), dim=0))
    num_pos = pos_feature.size(0)
    num_neg = neg_feature.size(0)
    dc_target = torch.from_numpy(np.array([[1]] * num_pos + [[0]] * num_neg)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)