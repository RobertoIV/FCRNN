import os, sys
import shutil
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from fcrn import FCRN
import torch
from utils import MetricLogger
import time
import datetime
from nyu_depth import NYUDepth
from config import config
from metrics import *
from tensorboardX import SummaryWriter
from tqdm import tqdm

def validate(dataloader, model, device, tb, epoch, tag):
    model.eval()
    with torch.no_grad():
        
        count = 0
        ratio_one = 0
        ratio_two = 0
        ratio_three = 0
        ard = 0
        srd = 0
        RMSE_linear = 0
        RMSE_log = 0
        RMSE_log_invariant = 0
        
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = [x.to(device) for x in data]
            image, depth_batch, normal_batch, conf_batch, cloud = data
            depth_pred_batch, normal_pred_batch = model(image)
            
            B = depth_pred_batch.size(0)
            count += B
            for i in range(B):
                pred = depth_pred_batch[i]
                depth = depth_batch[i]
                # only keep non-zero ones
                mask = depth > 1e-5
                pred = pred[mask]
                depth = depth[mask]
                # compute metrics
                ratio_one += ratio_correct(pred, depth, 1.25)
                ratio_two += ratio_correct(pred, depth, 1.25 ** 2)
                ratio_three += ratio_correct(pred, depth, 1.25 ** 3)
                
                ard += abs_rel_diff(pred, depth)
                srd += sq_rel_diff(pred, depth)
                
                RMSE_linear += rmse_linear(pred, depth)
                RMSE_log += rmse_log(pred, depth)
                RMSE_log_invariant += rmse_scale_invariant(pred, depth)
                

        ratio_one /= count
        ratio_two /= count
        ratio_three /= count
        ard /= count
        srd /= count
        RMSE_linear /= count
        RMSE_log /= count
        RMSE_log_invariant /= count

    if tag == 'test':
        min_depth = depth_batch[0].min()
        max_depth = depth_batch[0].max() * 1.25
        depth = (depth_batch[0] - min_depth) / (max_depth - min_depth)
        depth_pred = (depth_pred_batch[0] - min_depth) / (max_depth - min_depth)
        depth_pred = torch.clamp(depth_pred, min=0.0, max=1.0)
        normal = (normal_batch[0] + 1) / 2.0
        normal_pred = normal_pred_batch[0]
        normal_pred = (normal_pred + 1) / 2.0
        conf = conf_batch[0]
        
        tb.add_image('test/image', image[0], epoch)
        tb.add_image('test/depth', depth, epoch)
        tb.add_image('test/depth_pred', depth_pred, epoch)
        tb.add_image('test/normal', normal, epoch)
        tb.add_image('test/normal_pred', normal_pred, epoch)
        tb.add_image('test/conf', conf, epoch)
    
    tb.add_scalar('thres_1.25/' + tag, ratio_one, epoch)
    tb.add_scalar('thres_1.25_2/' + tag, ratio_two, epoch)
    tb.add_scalar('thres_1.25_3/' + tag, ratio_three, epoch)
    tb.add_scalar('ard/' + tag, ard, epoch)
    tb.add_scalar('srd/' + tag, srd, epoch)
    tb.add_scalar('rmse_linear/' + tag, RMSE_linear, epoch)
    tb.add_scalar('rmse_log/' + tag, RMSE_log, epoch)
    tb.add_scalar('rmse_log_invariant/' + tag, RMSE_log_invariant, epoch)
    
def main():

    resume = True
    path = 'data/NYU_DEPTH'
    batch_size = 32
    epochs = 1000
    device = torch.device('cuda:2')
    print_every = 5
    # exp_name = 'resnet18_nodropout_new'
    exp_name = 'consistency_soft_normal'
    lr = 1e-5
    weight_decay = 0.0005
    log_dir = os.path.join('logs', exp_name)
    model_dir = os.path.join('checkpoints', exp_name)
    val_every = 16
    save_every = 16


    # tensorboard
    # remove old log is not to resume
    if not resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tb = SummaryWriter(log_dir)
    tb.add_custom_scalars({
        'metrics': {
            'thres_1.25': ['Multiline', ['thres_1.25/train', 'thres_1.25/test']],
            'thres_1.25_2': ['Multiline', ['thres_1.25_2/train', 'thres_1.25_2/test']],
            'thres_1.25_3': ['Multiline', ['thres_1.25_3/train', 'thres_1.25_3/test']],
            'ard': ['Multiline', ['ard/train', 'ard/test']],
            'srd': ['Multiline', ['srd/train', 'srd/test']],
            'rmse_linear': ['Multiline', ['rmse_linear/train', 'rmse_linear/test']],
            'rmse_log': ['Multiline', ['rmse_log/train', 'rmse_log/test']],
            'rmse_log_invariant': ['Multiline', ['rmse_log_invariant/train', 'rmse_log_invariant/test']],
        }
    })
    
    
    # data loader
    dataset = NYUDepth(path, 'train')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    
    dataset_test = NYUDepth(path, 'test')
    dataloader_test = DataLoader(dataset_test, batch_size, shuffle=True, num_workers=4)
    
    
    # load model
    model = FCRN(True)
    model = model.to(device)
    
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    start_epoch = 0
    if resume:
        model_path = os.path.join(model_dir, 'model.pth')
        if os.path.exists(model_path):
            print('Loading checkpoint from {}...'.format(model_path))
            # load model and optimizer
            checkpoint = torch.load(os.path.join(model_dir, 'model.pth'))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('Model loaded.')
        else:
            print('No checkpoint found. Train from scratch')
    
    # training
    metric_logger = MetricLogger()
    
    end = time.perf_counter()
    max_iters = epochs * len(dataloader)
    
    def normal_loss(pred, normal, conf):
        """
        :param pred: (B, 3, H, W)
        :param normal: (B, 3, H, W)
        :param conf: 1
        """
        dot_prod = (pred * normal).sum(dim=1)
        # weighted loss, (B, )
        batch_loss = ((1 - dot_prod) * conf[:, 0]).sum(1).sum(1)
        # normalize, to (B, )
        batch_loss /= conf[:, 0].sum(1).sum(1)
        return batch_loss.mean()

    def consistency_loss(pred, cloud, normal, conf):
        """
        :param pred: (B, 1, H, W)
        :param normal: (B, 3, H, W)
        :param cloud: (B, 3, H, W)
        :param conf: (B, 1, H, W)
        """
        B, _, _, _ = normal.size()
        
        cloud = cloud.clone()
        cloud[:, 2:3, :, :] = pred
        # algorithm: use a kernel
        kernel = torch.ones((1, 1, 7, 7), device=pred.device)
        kernel = -kernel
        kernel[0, 0, 3, 3] = 48
    
        cloud_0 = cloud[:, 0:1]
        cloud_1 = cloud[:, 1:2]
        cloud_2 = cloud[:, 2:3]
        diff_0 = F.conv2d(cloud_0, kernel, padding=6, dilation=2)
        diff_1 = F.conv2d(cloud_1, kernel, padding=6, dilation=2)
        diff_2 = F.conv2d(cloud_2, kernel, padding=6, dilation=2)
        # (B, 3, H, W)
        diff = torch.cat((diff_0, diff_1, diff_2), dim=1)
        # normalize
        diff = F.normalize(diff, dim=1)
        # (B, 1, H, W)
        dot_prod = (diff * normal).sum(dim=1, keepdim=True)
        # weighted mean over image
        dot_prod = torch.abs(dot_prod.view(B, -1))
        conf = conf.view(B, -1)
        loss = (dot_prod * conf).sum(1) / conf.sum(1)
        # mean over batch
        return loss.mean()
    
    def criterion(depth_pred, normal_pred, depth, normal, cloud, conf):
        mse_loss = F.mse_loss(depth_pred, depth)
        consis_loss = consistency_loss(depth_pred, cloud, normal_pred, conf)
        norm_loss = normal_loss(normal_pred, normal, conf)
        
        return mse_loss, consis_loss, norm_loss
    
    print('Start training')
    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        for i, data in enumerate(dataloader):
            start = end
            i += 1
            data = [x.to(device) for x in data]
            image, depth, normal, conf, cloud = data
            depth_pred, normal_pred = model(image)
            mse_loss, consis_loss, norm_loss = criterion(depth_pred, normal_pred, depth, normal, cloud, conf)
            loss = mse_loss + consis_loss + norm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # bookkeeping
            end = time.perf_counter()
            metric_logger.update(loss=loss.item())
            metric_logger.update(mse_loss=mse_loss.item())
            metric_logger.update(norm_loss=norm_loss.item())
            metric_logger.update(consis_loss=consis_loss.item())
            metric_logger.update(batch_time=end-start)

            
            if i % print_every == 0:
                # Compute eta. global step: starting from 1
                global_step = epoch * len(dataloader) + i
                seconds = (max_iters - global_step) * metric_logger['batch_time'].global_avg
                eta = datetime.timedelta(seconds=int(seconds))
                # to display: eta, epoch, iteration, loss, batch_time
                display_dict = {
                    'eta': eta,
                    'epoch': epoch,
                    'iter': i,
                    'loss': metric_logger['loss'].median,
                    'batch_time': metric_logger['batch_time'].median
                }
                display_str = [
                    'eta: {eta}s',
                    'epoch: {epoch}',
                    'iter: {iter}',
                    'loss: {loss:.4f}',
                    'batch_time: {batch_time:.4f}s',
                ]
                print(', '.join(display_str).format(**display_dict))
                
                # tensorboard
                min_depth = depth[0].min()
                max_depth = depth[0].max() * 1.25
                depth = (depth[0] - min_depth) / (max_depth - min_depth)
                depth_pred = (depth_pred[0] - min_depth) / (max_depth - min_depth)
                depth_pred = torch.clamp(depth_pred, min=0.0, max=1.0)
                normal = (normal[0] + 1) / 2
                normal_pred = (normal_pred[0] + 1) / 2
                conf = conf[0]
                
                tb.add_scalar('train/loss', metric_logger['loss'].median, global_step)
                tb.add_scalar('train/mse_loss', metric_logger['mse_loss'].median, global_step)
                tb.add_scalar('train/consis_loss', metric_logger['consis_loss'].median, global_step)
                tb.add_scalar('train/norm_loss', metric_logger['norm_loss'].median, global_step)
                
                tb.add_image('train/depth', depth, global_step)
                tb.add_image('train/normal', normal, global_step)
                tb.add_image('train/depth_pred', depth_pred, global_step)
                tb.add_image('train/normal_pred', normal_pred, global_step)
                tb.add_image('train/conf', conf, global_step)
                tb.add_image('train/image', image[0], global_step)
                
        if (epoch) % val_every == 0:
            # validate after each epoch
            validate(dataloader, model, device, tb, epoch, 'train')
            validate(dataloader_test, model, device, tb, epoch, 'test')
        if (epoch) % save_every == 0:
            to_save = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'epoch': epoch,
            }
            torch.save(to_save, os.path.join(model_dir, 'model.pth'))


if __name__ == '__main__':
    main()
