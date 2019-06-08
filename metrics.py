import torch

"""
For all the following, the input is just a flattened array.
We sum over batch
"""

def ratio_correct(pred, target, threshold):
    """
    :param pred: (B, L)
    :param target: (B, L)
    :param threshold: like 1.25
    :param mask: (B, L), valid mask
    """
    ratio = torch.max(pred / (target + 1e-6), target / (pred + 1e-6))
    correct = ratio <= threshold
    ratio_correct = correct.float().mean()
    
    return ratio_correct.item()

def abs_rel_diff(pred, target):
    """
    :param pred: (B, L)
    :param target: (B, L)
    :param mask: (B, L), valid mask
    """
    ard = torch.abs(pred - target) / (target + 1e-6)
    return ard.mean().item()

def sq_rel_diff(pred, target):
    """
    :param pred: (B, L)
    :param target: (B, L)
    :param mask: (B, L), valid mask
    """
    srd = torch.pow(pred - target, 2) / (target + 1e-6)
    return srd.mean().item()

def rmse_linear(pred, target):
    """
    :param pred: (B, L)
    :param target: (B, L)
    :param mask: (B, L), valid mask
    """
    rmse = torch.sqrt(torch.pow(pred - target, 2).mean())
    return rmse.mean().item()

def rmse_log(pred, target):
    """
    :param pred: (B, L)
    :param target: (B, L)
    :param mask: (B, L), valid mask
    """
    pred, target = torch.log(pred + 1e-6), torch.log(target + 1e-6)
    rmse = torch.sqrt(torch.pow(pred - target, 2).mean())
    return rmse.item()

def rmse_scale_invariant(pred, target):
    """
    :param pred: (B, L)
    :param target: (B, L)
    :param mask: (B, L), valid mask
    """
    pred, target = torch.log(pred + 1e-6), torch.log(target + 1e-6)
    
    # alpha = (target - pred).mean()
    # rmse = torch.pow(pred - target - alpha, 2).mean() / 2
    n = pred.size()[0]
    d = pred - target
    rmse = (d ** 2).sum() / n - (d.sum()) ** 2 / n ** 2
    
    return rmse.item()
    
    
if __name__ == '__main__':
    a = torch.rand(1, 1, 3, 3)
    b = torch.rand(1, 1, 3, 3)
    # b = torch.ones(1, 3, 3).float()
    mask = b > 1e-3
    a = a[mask]
    b = b[mask]
    
    ratio_one = ratio_correct(a, b, 1.25)
    ratio_two = ratio_correct(a, b, 1.25 ** 2)
    ratio_three = ratio_correct(a, b, 1.25 ** 3)
    
    ard = abs_rel_diff(a, b)
    srd = sq_rel_diff(a, b)
    
    RMSE_linear = rmse_linear(a, b)
    RMSE_log = rmse_log(a, b)
    RMSE_invariant = rmse_scale_invariant(a, b)
    
    print(ratio_one)
    print(ratio_two)
    print(ratio_three)
    print(ard)
    print(srd)
    print(RMSE_linear)
    print(RMSE_log)
    print(RMSE_invariant)

    import numpy as np
    input_gt_depth_image = b[0].numpy()
    pred_depth_image = a[0].numpy()
    n = np.sum(input_gt_depth_image > 1e-3)

    idxs = (input_gt_depth_image <= 1e-3)
    pred_depth_image[idxs] = 1
    input_gt_depth_image[idxs] = 1

    pred_d_gt = pred_depth_image / input_gt_depth_image
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth_image / pred_depth_image
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n

    log_pred = np.log(pred_depth_image)
    log_gt = np.log(input_gt_depth_image)

    d_i = log_gt - log_pred

    RMSE_linear = np.sqrt(np.sum((pred_depth_image - input_gt_depth_image) ** 2) / n)
    RMSE_log = np.sqrt(np.sum((log_pred - log_gt) ** 2) / n)
    RMSE_log_scale_invariant = np.sum(d_i ** 2) / n + (np.sum(d_i) ** 2) / (n ** 2)
    ARD = np.sum(np.abs((pred_depth_image - input_gt_depth_image)) / input_gt_depth_image) / n
    SRD = np.sum(((pred_depth_image - input_gt_depth_image) ** 2) / input_gt_depth_image) / n

    print('Threshold_1_25: {}'.format(Threshold_1_25))
    print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
    print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
    print('RMSE_linear: {}'.format(RMSE_linear))
    print('RMSE_log: {}'.format(RMSE_log))
    print('RMSE_log_scale_invariant: {}'.format(RMSE_log_scale_invariant))
    print('ARD: {}'.format(ARD))
    print('SRD: {}'.format(SRD))
    
