import os, sys
import shutil
from torch.utils.data import DataLoader
from torch import nn, optim
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
            image, depth_batch, cloud = data
            pred_batch = model(image)
            
            B = pred_batch.size(0)
            count += B
            for i in range(B):
                pred = pred_batch[i]
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
        pred = (pred_batch[0] - min_depth) / (max_depth - min_depth)
        pred = torch.clamp(pred, min=0.0, max=1.0)
        tb.add_image('test/image', image[0], epoch)
        tb.add_image('test/depth', depth, epoch)
        tb.add_image('test/pred', pred, epoch)
    
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
    device = torch.device('cuda:3')
    print_every = 5
    exp_name = 'resnet18_nodropout_new'
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
        print('Loading checkpoint from {}...'.format(os.path.join(model_dir, 'model.pth')))
        # load model and optimizer
        checkpoint = torch.load(os.path.join(model_dir, 'model.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Model loaded.')
    
    criterion = nn.MSELoss()
    # training
    metric_logger = MetricLogger()
    
    end = time.perf_counter()
    max_iters = epochs * len(dataloader)
    
    print('Start training')
    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        for i, data in enumerate(dataloader):
            start = end
            i += 1
            data = [x.to(device) for x in data]
            image, depth, cloud = data
            pred = model(image)
            loss = criterion(pred, depth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # bookkeeping
            end = time.perf_counter()
            metric_logger.update(loss=loss.item())
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
                pred = (pred[0] - min_depth) / (max_depth - min_depth)
                pred = torch.clamp(pred, min=0.0, max=1.0)
                tb.add_scalar('train/loss', loss.item(), global_step)
                tb.add_image('train/depth', depth, global_step)
                tb.add_image('train/pred', pred, global_step)
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
