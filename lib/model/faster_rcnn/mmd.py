import torch
from functools import partial
from torch.autograd import Variable


# Consider linear time KD_MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def mmd_rbf_noaccelerate_bary(one_pts, all_pts, kernel_mul=2.0, kernel_num=5, fix_sigma=None, start_at_zero=False):
    one_pts = one_pts.view(1, -1)
    batch_size = int(one_pts.size()[0])
    num_domains_handled = all_pts.shape[0]
    start_index = 0
    if not start_at_zero:
        num_domains_handled -= 1
        start_index = 1

    XX = None
    XXI_sum = None
    XIXJ_sum = None

    for i in range(start_index, all_pts.shape[0]):
        one_pt_unsqueeze = all_pts[i].view(1, -1)

        kernels_1 = guassian_kernel(one_pts, one_pt_unsqueeze,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

        if (one_pts - one_pt_unsqueeze).sum() == 0:
            if XXI_sum is None:
                XXI_sum = 1
            else:
                XXI_sum += 1
        else:
            if XXI_sum is None:
                XXI_sum = kernels_1[:batch_size, batch_size:]
            else:
                XXI_sum += kernels_1[:batch_size, batch_size:]
            if XX is None:
                XX = kernels_1[:batch_size, :batch_size]


        for j in range(start_index, all_pts.shape[0]):
            #aone_pt_unsqueeze = all_pts[j].unsqueeze(0)  # To get correct format
            aone_pt_unsqueeze = all_pts[j].view(1, -1)

            kernels_2 = guassian_kernel(one_pt_unsqueeze, aone_pt_unsqueeze,
                                      kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

            if i == j:
                if XIXJ_sum is None:
                    XIXJ_sum = 1
                else:
                    XIXJ_sum += 1
            else:
                if XIXJ_sum is None:
                    XIXJ_sum = kernels_2[batch_size:, :batch_size]
                else:
                    XIXJ_sum += kernels_2[batch_size:, :batch_size]
    mmd_loss = XX - (2/num_domains_handled) * XXI_sum + (1/(num_domains_handled**2))*XIXJ_sum
    return mmd_loss#.sqrt()

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)
    return output

def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)
    return torch.sum(torch.exp(-s), 0).view_as(dist)
def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    return cost

def maximum_mean_discrepancy_bary(all, current_i, kernel= gaussian_kernel_matrix, include_current=False):
    XX = torch.mean(kernel(all[current_i].view(1, -1), all[current_i].view(1, -1)))
    n_samples = len(all)

    start = 1
    if include_current:
        start = 0

    XXI = None
    XJXI = None
    for i in range(start, n_samples):
        if XXI is None:
            XXI = torch.mean(kernel(all[current_i].view(1, -1), all[i].view(1, -1)))
        else:
            XXI = XXI + torch.mean(kernel(all[current_i].view(1, -1), all[i].view(1, -1)))

        for j in range(start, n_samples):
            if XJXI is None:
                XJXI = torch.mean(kernel(all[i].view(1, -1), all[j].view(1, -1)))
            else:
                XJXI = XJXI + torch.mean(kernel(all[i].view(1, -1), all[j].view(1, -1)))


    cost = XX + (2/n_samples) * XXI + (1/n_samples**2) * XJXI
    return cost.sqrt()

def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value
    return loss_value

def mmd_loss_bary(all, current_feature_i, include_current=False):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy_bary(all, current_feature_i, kernel=gaussian_kernel, include_current=include_current)
    return loss_value



